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
SHOW_OBSERVED = True  # faint black observed history per pedestrian
SHOW_OBSTACLES = True  # red X markers — obstacles the controller used
AXIS_PAD = 1.5  # meters of padding around the data bounding box


def compute_axis_limits(start, goal, ego_positions, ped_positions, ped_mask,
                        predicted_positions, observed_positions):
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
    if observed_positions.size:
        xs.extend(observed_positions.reshape(-1, 2)[:, 0].tolist())
        ys.extend(observed_positions.reshape(-1, 2)[:, 1].tolist())
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
    observed_positions = data["pedestrian_observed"]  # (P, OBSERVE_LEN, 2)
    predicted_positions = data["predicted_positions"]  # (P, H, 2)
    obstacle_positions = data["obstacle_positions"]  # (T, P, 2)
    obstacle_mask = data["obstacle_mask"]  # (T, P)

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

    num_frames = ego_positions.shape[0]  # T + 1
    num_peds = ped_positions.shape[1]  # display axis (all peds in scene)
    num_anchor_peds = observed_positions.shape[0]  # anchor cohort axis
    horizon = predicted_positions.shape[1]

    print(f"Loaded rollout: {num_frames} frames, {num_peds} pedestrians on "
          f"display, {num_anchor_peds} with full history, horizon {horizon}")

    xlim, ylim = compute_axis_limits(
        start, goal, ego_positions, ped_positions, ped_mask,
        predicted_positions, observed_positions if SHOW_OBSERVED else np.zeros((0, 0, 2)),
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

    # Static items: start, goal, and (optionally) observed histories.
    ax.plot(start[0], start[1], "bs", markersize=10, label="Start")
    ax.plot(goal[0],  goal[1],  "b*", markersize=16, label="Goal")

    if SHOW_OBSERVED:
        for i in range(num_anchor_peds):
            hist = observed_positions[i]
            ax.plot(hist[:, 0], hist[:, 1], "-",
                    color="0.6", linewidth=0.8, alpha=0.5,
                    label="Observed history" if i == 0 else None)

    # Dynamic artists, updated each frame.
    ego_path_line, = ax.plot([], [], "b-", linewidth=2.0, label="Ego path")
    ego_dot,        = ax.plot([], [], "bo", markersize=11, label="Ego")
    peds_dots,      = ax.plot([], [], "ko", markersize=6,  label="Pedestrians")

    pred_lines = []
    if SHOW_PREDICTIONS:
        for i in range(num_anchor_peds):
            line, = ax.plot([], [], "r--", linewidth=0.9, alpha=0.55,
                            label="Predicted future" if i == 0 else None)
            pred_lines.append(line)

    obs_dots = None
    if SHOW_OBSTACLES:
        obs_dots, = ax.plot([], [], "rx", markersize=8, alpha=0.8,
                            label="Obstacle (controller)")

    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")
    time_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top",
                        fontsize=10, family="monospace")

    ax.legend(loc="lower right", fontsize=8)

    def update(t):
        # Ego: path up to and including step t, plus a marker at current pos.
        ego_path_line.set_data(ego_positions[:t + 1, 0], ego_positions[:t + 1, 1])
        ego_dot.set_data([ego_positions[t, 0]], [ego_positions[t, 1]])

        # Pedestrians visible at this step.
        m = ped_mask[t]
        peds_dots.set_data(ped_positions[t, m, 0], ped_positions[t, m, 1])

        # Predicted future: predictions were made once at anchor (t=0), so
        # at sim step t the still-relevant horizon is offsets [t..H-1].
        if SHOW_PREDICTIONS:
            for i, line in enumerate(pred_lines):
                if t < horizon:
                    line.set_data(
                        predicted_positions[i, t:, 0],
                        predicted_positions[i, t:, 1],
                    )
                else:
                    line.set_data([], [])

        # Obstacle markers used during the t-th control step. obstacle_*
        # arrays are indexed 0..T-1 (one shorter than ego_positions), so the
        # final frame just shows the ego at the goal with no obstacle markers.
        if obs_dots is not None:
            if t < obstacle_positions.shape[0]:
                mo = obstacle_mask[t]
                obs_dots.set_data(
                    obstacle_positions[t, mo, 0],
                    obstacle_positions[t, mo, 1],
                )
            else:
                obs_dots.set_data([], [])

        dt = metadata.get("dt_seconds", 0.4)
        step_text.set_text(f"step {t}/{num_frames - 1}")
        time_text.set_text(f"t = {t * dt:.1f} s")

        artists = [ego_path_line, ego_dot, peds_dots, step_text, time_text]
        artists.extend(pred_lines)
        if obs_dots is not None:
            artists.append(obs_dots)
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / FPS, blit=False, repeat=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_animation(anim, FPS)


if __name__ == "__main__":
    main()
