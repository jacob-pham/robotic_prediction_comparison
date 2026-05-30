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
SHOW_PREDICTIONS = True
AXIS_PAD = 1.5


def compute_axis_limits(start, goal, ego_positions, ped_positions, ped_mask,
                        step_predicted_positions, step_predicted_mask):
    """Compute x/y axis limits that fit everything in the scene with padding.

    input:
        start: (2,) ego start position
        goal: (2,) ego goal position
        ego_positions: (T+1, 2) ego path
        ped_positions: (T+1, P, 2) pedestrian positions over time
        ped_mask: (T+1, P) bool, True where a pedestrian is present
        step_predicted_positions: (T, P_obs, H, 2) predicted futures per step
        step_predicted_mask: (T, P_obs) bool, True where a prediction exists
    output:
        xlim: (x_min, x_max) with AXIS_PAD applied
        ylim: (y_min, y_max) with AXIS_PAD applied
    """
    # start with the ego path and the start/goal markers
    all_points = [np.array([start, goal]), ego_positions]

    # add all pedestrian positions that are actually present
    valid_peds = ped_positions[ped_mask]
    if valid_peds.size > 0:
        all_points.append(valid_peds)

    # add all predicted future positions that are valid
    num_pred_steps = step_predicted_positions.shape[0]
    num_obs_peds   = step_predicted_positions.shape[1]
    for t in range(num_pred_steps):
        for j in range(num_obs_peds):
            if step_predicted_mask[t, j]:
                all_points.append(step_predicted_positions[t, j])

    all_xy = np.concatenate(all_points, axis=0)
    xlim = (all_xy[:, 0].min() - AXIS_PAD, all_xy[:, 0].max() + AXIS_PAD)
    ylim = (all_xy[:, 1].min() - AXIS_PAD, all_xy[:, 1].max() + AXIS_PAD)
    return xlim, ylim


def main():
    """Load the saved rollout file and write an animation to disk.

    input:
        None (reads RESULTS_FILE and METADATA_FILE set at the top of this file)
    output:
        None (writes a .mp4 or .gif next to the rollout file)
    """
    if not RESULTS_FILE.exists():
        print(f"ERROR: no rollout file at {RESULTS_FILE}")
        print("Run the simulation first:  python -m navigation.simulate_simple")
        sys.exit(1)

    # load the rollout data
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
    num_pred_steps = step_predicted_positions.shape[0]
    horizon = step_predicted_positions.shape[2]

    print(f"Loaded rollout: {num_frames} frames, {num_peds} peds, horizon {horizon}")

    scene = metadata.get("scene_name", "?")
    anchor = metadata.get("anchor_frame", "?")
    dt = metadata.get("dt_seconds", 0.4)

    xlim, ylim = compute_axis_limits(
        start, goal, ego_positions,
        ped_positions, ped_mask,
        step_predicted_positions, step_predicted_mask,
    )

    # set up the figure
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Navigation rollout - {scene} (anchor frame {anchor})")
    ax.grid(True, alpha=0.3)

    # static markers for start and goal
    ax.plot(start[0], start[1], "bs", markersize=10, label="Start")
    ax.plot(goal[0],  goal[1],  "b*", markersize=16, label="Goal")

    # dynamic artists that get updated each frame
    ego_path_line, = ax.plot([], [], "b-", linewidth=2.0, label="Ego path")
    ego_dot, = ax.plot([], [], "bo", markersize=11, label="Ego")
    peds_dots, = ax.plot([], [], "ko", markersize=6, label="Pedestrians")

    # one dashed red line per obstacle pedestrian for predicted futures
    pred_lines = []
    if SHOW_PREDICTIONS:
        for i in range(num_obs_peds):
            label = "Predicted future" if i == 0 else None
            line, = ax.plot([], [], "--r", marker="o", markersize=3,
                            linewidth=0.9, alpha=0.6, label=label)
            pred_lines.append(line)

    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        va="top", fontsize=10, family="monospace")
    time_text = ax.text(0.02, 0.91, "", transform=ax.transAxes,
                        va="top", fontsize=10, family="monospace")
    ax.legend(loc="lower right", fontsize=8)

    # animation update function
    def update(t):
        """Draw one frame of the animation.

        input:
            t: current time step index
        output:
            list of matplotlib artists that were updated
        """
        # update ego trail and current position dot
        ego_path_line.set_data(ego_positions[:t + 1, 0], ego_positions[:t + 1, 1])
        ego_dot.set_data([ego_positions[t, 0]], [ego_positions[t, 1]])

        # show only pedestrians present at this step
        visible = ped_positions[t][ped_mask[t]]
        if len(visible) > 0:
            peds_dots.set_data(visible[:, 0], visible[:, 1])
        else:
            peds_dots.set_data([], [])

        # update predicted future lines (blank them on the last frame)
        if SHOW_PREDICTIONS:
            if t < num_pred_steps:
                for i, line in enumerate(pred_lines):
                    if step_predicted_mask[t, i]:
                        line.set_data(step_predicted_positions[t, i, :, 0],
                                      step_predicted_positions[t, i, :, 1])
                    else:
                        line.set_data([], [])
            else:
                for line in pred_lines:
                    line.set_data([], [])

        step_text.set_text(f"step {t}/{num_frames - 1}")
        time_text.set_text(f"t = {t * dt:.1f} s")

        return [ego_path_line, ego_dot, peds_dots, step_text, time_text] + pred_lines

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / FPS, blit=False, repeat=False,
    )

    # save the animation - try mp4 first, fall back to gif if ffmpeg isn't available
    if animation.writers.is_available("ffmpeg"):
        try:
            writer = animation.FFMpegWriter(fps=FPS, bitrate=BITRATE)
            anim.save(OUTPUT_MP4, writer=writer, dpi=DPI)
            print(f"Saved animation -> {OUTPUT_MP4}")
            return
        except Exception as exc:
            print(f"MP4 save failed ({exc}); falling back to GIF.")

    try:
        anim.save(OUTPUT_GIF, writer="pillow", fps=FPS, dpi=DPI)
        print(f"Saved animation -> {OUTPUT_GIF}")
    except Exception as exc:
        print(f"Animation save failed: {exc}")
        print("Install either ffmpeg (for .mp4) or pillow (for .gif).")


if __name__ == "__main__":
    main()
