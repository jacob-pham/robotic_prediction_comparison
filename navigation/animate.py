"""Animate a saved navigation rollout.

Run from project root:  python -m navigation.animate
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# use the ffmpeg bundled with imageio-ffmpeg; the system one can fail on Windows
try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# optional CLI arg: path to a different .npz rollout
if len(sys.argv) > 1:
    RESULTS_FILE = Path(sys.argv[1]).resolve()
else:
    RESULTS_FILE = RESULTS_DIR / "navigation_rollout_simple.npz"
METADATA_FILE = RESULTS_FILE.with_name(RESULTS_FILE.stem + "_metadata.json")
OUTPUT_GIF = RESULTS_FILE.with_suffix(".gif")
OUTPUT_MP4 = RESULTS_FILE.with_suffix(".mp4")

FPS = 6
BITRATE = 1800
DPI = 120
SHOW_PREDICTIONS = True  # red dashed prediction line per obstacle
AXIS_PAD = 1.5  # padding around the data bounding box (meters)


def compute_axis_limits(start, goal, ego_positions, ped_positions, ped_mask,
                        predicted_positions):
    """Bounding box around everything plotted, with padding.

    input:
        start: (2,) ego start
        goal: (2,) ego goal
        ego_positions: (T+1, 2) ego path
        ped_positions: (T+1, P, 2) per-step ped positions (NaN allowed)
        ped_mask: (T+1, P) bool mask for ped_positions
        predicted_positions: (N, 2) flat array of valid predicted points
    output:
        (xlim, ylim) tuples of (min, max) each with AXIS_PAD applied
    """
    xs = []
    ys = []

    # always include the start and goal markers
    xs.append(start[0])
    ys.append(start[1])
    xs.append(goal[0])
    ys.append(goal[1])

    # include the whole ego path
    for t in range(ego_positions.shape[0]):
        xs.append(ego_positions[t, 0])
        ys.append(ego_positions[t, 1])

    # include every pedestrian position that is actually present
    num_frames = ped_positions.shape[0]
    num_peds = ped_positions.shape[1]
    for t in range(num_frames):
        for j in range(num_peds):
            if ped_mask[t, j]:
                xs.append(ped_positions[t, j, 0])
                ys.append(ped_positions[t, j, 1])

    # include every predicted point (already flattened to (N, 2))
    flat_preds = predicted_positions.reshape(-1, 2)
    for i in range(flat_preds.shape[0]):
        xs.append(flat_preds[i, 0])
        ys.append(flat_preds[i, 1])

    xlim = (min(xs) - AXIS_PAD, max(xs) + AXIS_PAD)
    ylim = (min(ys) - AXIS_PAD, max(ys) + AXIS_PAD)
    return xlim, ylim


def save_animation(anim, fps):
    """Write the animation, MP4 if ffmpeg is available else GIF.

    input:
        anim: matplotlib FuncAnimation
        fps: frames per second
    output:
        None (writes a video next to RESULTS_FILE)
    """
    if animation.writers.is_available("ffmpeg"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=BITRATE)
            anim.save(OUTPUT_MP4, writer=writer, dpi=DPI)
            print(f"Saved animation -> {OUTPUT_MP4}")
            return
        except Exception as exc:
            print(f"MP4 save failed ({exc}); falling back to GIF.")

    try:
        anim.save(OUTPUT_GIF, writer="pillow", fps=fps, dpi=DPI)
        print(f"Saved animation -> {OUTPUT_GIF}")
    except Exception as exc:
        print(f"Animation save failed: {exc}")
        print("Install either ffmpeg (for .mp4) or pillow (for .gif).")


def main():
    """Load the saved rollout and write an animation.

    input:
        None (reads RESULTS_FILE / METADATA_FILE)
    output:
        None (writes a .mp4 or .gif next to the rollout file)
    """
    if not RESULTS_FILE.exists():
        print(f"ERROR: no rollout file at {RESULTS_FILE}")
        print("Run the simulation first:  python -m navigation.simulate_simple")
        sys.exit(1)

    data = np.load(RESULTS_FILE, allow_pickle=False)
    ego_positions = data["ego_positions"]  # (T+1, 2)
    start = data["start"]  # (2,)
    goal = data["goal"]  # (2,)
    ped_positions = data["pedestrian_positions"]  # (T+1, P, 2)
    ped_mask = data["pedestrian_mask"]  # (T+1, P)
    step_predicted_positions = data["step_predicted_positions"]  # (T, P_obs, H, 2)
    step_predicted_mask = data["step_predicted_mask"]  # (T, P_obs)

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

    num_frames = ego_positions.shape[0]
    num_peds = ped_positions.shape[1]
    num_obs_peds = step_predicted_positions.shape[1]
    horizon = step_predicted_positions.shape[2]

    print(f"Loaded rollout: {num_frames} frames, {num_peds} pedestrians on "
          f"display, {num_obs_peds} obstacle peds, horizon {horizon}")

    # bbox: gather every valid predicted point too
    valid_pred_list = []
    num_pred_steps = step_predicted_positions.shape[0]
    for t in range(num_pred_steps):
        for j in range(num_obs_peds):
            if step_predicted_mask[t, j]:
                for h in range(horizon):
                    valid_pred_list.append(step_predicted_positions[t, j, h])
    if valid_pred_list:
        valid_preds = np.array(valid_pred_list, dtype=np.float32)
    else:
        valid_preds = np.zeros((0, 2), dtype=np.float32)
    xlim, ylim = compute_axis_limits(
        start, goal, ego_positions, ped_positions, ped_mask, valid_preds,
    )

    scene = metadata.get("scene_name", "?")
    anchor = metadata.get("anchor_frame", "?")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Navigation rollout - {scene} (anchor frame {anchor})")
    ax.grid(True, alpha=0.3)

    # static start/goal markers
    ax.plot(start[0], start[1], "bs", markersize=10, label="Start")
    ax.plot(goal[0],  goal[1],  "b*", markersize=16, label="Goal")

    # dynamic artists updated each frame (ax.plot returns a list, take item 0)
    ego_path_line = ax.plot([], [], "b-", linewidth=2.0, label="Ego path")[0]
    ego_dot = ax.plot([], [], "bo", markersize=11, label="Ego")[0]
    peds_dots = ax.plot([], [], "ko", markersize=6, label="Pedestrians")[0]

    # one prediction line per obstacle slot, blanked when no prediction
    pred_lines = []
    if SHOW_PREDICTIONS:
        for i in range(num_obs_peds):
            # only label the first line so the legend shows one entry
            if i == 0:
                label = "Predicted future"
            else:
                label = None
            line = ax.plot([], [], linestyle="--", color="red",
                           marker="o", markersize=3,
                           linewidth=0.9, alpha=0.6, label=label)[0]
            pred_lines.append(line)

    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")
    time_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")

    ax.legend(loc="lower right", fontsize=8)

    def update(t):
        """Draw one animation frame.

        input:
            t: sim step index (0..num_frames-1)
        output:
            list of artists that were updated (for matplotlib)
        """
        # ego: trail + current dot
        ego_path_line.set_data(ego_positions[:t + 1, 0], ego_positions[:t + 1, 1])
        ego_dot.set_data([ego_positions[t, 0]], [ego_positions[t, 1]])

        # peds visible at this step
        visible_x = []
        visible_y = []
        for j in range(num_peds):
            if ped_mask[t, j]:
                visible_x.append(ped_positions[t, j, 0])
                visible_y.append(ped_positions[t, j, 1])
        peds_dots.set_data(visible_x, visible_y)

        # predicted futures (last frame has no controller step, so blank)
        if SHOW_PREDICTIONS:
            if t < num_pred_steps:
                for i, line in enumerate(pred_lines):
                    if step_predicted_mask[t, i]:
                        line.set_data(
                            step_predicted_positions[t, i, :, 0],
                            step_predicted_positions[t, i, :, 1],
                        )
                    else:
                        line.set_data([], [])
            else:
                for line in pred_lines:
                    line.set_data([], [])

        dt = metadata.get("dt_seconds", 0.4)
        step_text.set_text(f"step {t}/{num_frames - 1}")
        time_text.set_text(f"t = {t * dt:.1f} s")

        artists = [ego_path_line, ego_dot, peds_dots, step_text, time_text]
        artists.extend(pred_lines)
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / FPS, blit=False, repeat=False,
    )

    save_animation(anim, FPS)


if __name__ == "__main__":
    main()
