"""Animate a saved navigation rollout.

Loads navigation/results/navigation_rollout_simple.npz (and metadata JSON if
present) and writes an animation to the same folder. Does NOT touch the SSM
model, the raw dataset, or the potential-field controller — everything needed
is in the saved rollout.

Run from project root:   python -m navigation.animate   (or  python navigation/animate.py)
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Point matplotlib at the ffmpeg binary bundled with imageio-ffmpeg. The
# system ffmpeg on Windows can fail with a DLL error; the bundled one is a
# standalone static binary and just works.
try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Optional first CLI arg: path to a different .npz rollout. Default is the
# output of simulate_simple.py.
if len(sys.argv) > 1:
    RESULTS_FILE = Path(sys.argv[1]).resolve()
else:
    RESULTS_FILE = RESULTS_DIR / "navigation_rollout_simple.npz"
METADATA_FILE = RESULTS_FILE.with_name(RESULTS_FILE.stem + "_metadata.json")
OUTPUT_GIF = RESULTS_FILE.with_suffix(".gif")
OUTPUT_MP4 = RESULTS_FILE.with_suffix(".mp4")

# Visual options.
FPS = 6
BITRATE = 1800  # mp4 bitrate
DPI = 120  # render dpi for the saved animation
SHOW_PREDICTIONS = True  # red dashed line — predicted future from current step
AXIS_PAD = 1.5  # meters of padding around the data bounding box


def compute_axis_limits(start, goal, ego_positions, ped_positions, ped_mask,
                        predicted_positions):
    """Bounding box around everything we plot, plus padding."""
    xs = [start[0], goal[0]]
    ys = [start[1], goal[1]]
    xs.extend(ego_positions[:, 0].tolist())
    ys.extend(ego_positions[:, 1].tolist())
    if ped_mask.any():
        xs.extend(ped_positions[ped_mask][:, 0].tolist())
        ys.extend(ped_positions[ped_mask][:, 1].tolist())
    if predicted_positions.size:
        xs.extend(predicted_positions.reshape(-1, 2)[:, 0].tolist())
        ys.extend(predicted_positions.reshape(-1, 2)[:, 1].tolist())
    return (min(xs) - AXIS_PAD, max(xs) + AXIS_PAD), (min(ys) - AXIS_PAD, max(ys) + AXIS_PAD)


def save_animation(anim, fps):
    """Try MP4 (ffmpeg), fall back to GIF (pillow). Print a helpful message
    if neither is available."""
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

    num_frames = ego_positions.shape[0]  # T + 1
    num_peds = ped_positions.shape[1]  # display axis (all peds in scene)
    num_obs_peds = step_predicted_positions.shape[1]  # union of peds seen as obstacles
    horizon = step_predicted_positions.shape[2]

    print(f"Loaded rollout: {num_frames} frames, {num_peds} pedestrians on "
          f"display, {num_obs_peds} obstacle peds, horizon {horizon}")

    # Bounding box: include any predicted point from any step (NaN-safe).
    if step_predicted_mask.any():
        valid_preds = step_predicted_positions[step_predicted_mask]
    else:
        valid_preds = np.zeros((0, 2), dtype=np.float32)
    xlim, ylim = compute_axis_limits(
        start, goal, ego_positions, ped_positions, ped_mask, valid_preds,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    scene = metadata.get("scene_name", "?")
    anchor = metadata.get("anchor_frame", "?")
    ax.set_title(f"Navigation rollout - {scene} (anchor frame {anchor})")
    ax.grid(True, alpha=0.3)

    # Static items: start and goal.
    ax.plot(start[0], start[1], "bs", markersize=10, label="Start")
    ax.plot(goal[0],  goal[1],  "b*", markersize=16, label="Goal")

    # Dynamic artists, updated each frame.
    ego_path_line, = ax.plot([], [], "b-", linewidth=2.0, label="Ego path")
    ego_dot,        = ax.plot([], [], "bo", markersize=11, label="Ego")
    peds_dots,      = ax.plot([], [], "ko", markersize=6,  label="Pedestrians")

    # One line per obstacle slot — we update its data each frame from the
    # per-step predictions saved by the simulator. Slots that have no
    # prediction at a given step are blanked out.
    # Dashed segments connect successive predicted steps, and a marker at
    # each vertex makes it easy to count off the 8 future positions.
    pred_lines = []
    if SHOW_PREDICTIONS:
        for i in range(num_obs_peds):
            line, = ax.plot([], [], linestyle="--", color="red",
                            marker="o", markersize=3,
                            linewidth=0.9, alpha=0.6,
                            label="Predicted future" if i == 0 else None)
            pred_lines.append(line)

    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")
    time_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")

    ax.legend(loc="lower right", fontsize=8)

    num_pred_steps = step_predicted_positions.shape[0]  # T (one shorter than ego_positions)

    def update(t):
        # Ego: path up to and including step t, plus a marker at current pos.
        ego_path_line.set_data(ego_positions[:t + 1, 0], ego_positions[:t + 1, 1])
        ego_dot.set_data([ego_positions[t, 0]], [ego_positions[t, 1]])

        # Pedestrians visible at this step.
        m = ped_mask[t]
        peds_dots.set_data(ped_positions[t, m, 0], ped_positions[t, m, 1])

        # Predicted future: re-predicted at every control step. Index t into
        # the per-step predictions; the last frame (t == num_pred_steps) has
        # no controller action, so we just blank the prediction lines.
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_animation(anim, FPS)


if __name__ == "__main__":
    main()
