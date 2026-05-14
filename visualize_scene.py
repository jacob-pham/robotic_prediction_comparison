"""Animate a full ETH/UCY scene from one raw .txt file.

Loads the raw pedestrian data, groups it by frame and by pedestrian,
and saves an animation showing everyone walking through the scene.
Also saves a static plot of all trajectories.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import tqdm

# Use the ffmpeg binary bundled with imageio-ffmpeg. The system ffmpeg on
# Windows can fail with a DLL error; the bundled one is a standalone static
# binary and just works. (Same trick as navigation/animate.py.)
try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


# Path to the raw .txt file. Change this to view a different scene.
RAW_FILE = Path("datasets") / "raw" / "all_data" / "students003.txt"

# Output files
ANIMATION_OUT = Path("datasets_processed") / "univ" / "scene_animation_students003.mp4"  # falls back to .gif if ffmpeg missing
STATIC_OUT = Path("datasets_processed") / "univ" / "scene_full_tracks_students003.png"

# Animation controls
FRAME_STRIDE = 1  # show every Nth frame (set higher to speed up)
MAX_FRAMES = None  # cap the number of animated frames, or None
TRAIL_LENGTH = 20  # how many past frames of trail to draw per ped
SHOW_PED_IDS = False  # annotate each point with the pedestrian id
FPS = 10  # animation playback speed
MARGIN = 1.0  # padding around min/max of scene (meters)


def load_scene(path):
    """Read a raw .txt file with columns [frame_id, ped_id, x, y]."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["frame_id", "ped_id", "x", "y"],
    )
    df = df.sort_values("frame_id").reset_index(drop=True)
    return df


def group_by_frame(df):
    """Return a dict: frame_id -> list of (ped_id, x, y) for that frame."""
    frames = {}
    for frame_id, group in df.groupby("frame_id"):
        peds = list(zip(group["ped_id"].values,
                        group["x"].values,
                        group["y"].values))
        frames[int(frame_id)] = peds
    return frames


def group_by_ped(df):
    """Return a dict: ped_id -> list of (frame_id, x, y) sorted by frame."""
    peds = {}
    for ped_id, group in df.groupby("ped_id"):
        group = group.sort_values("frame_id")
        peds[ped_id] = list(zip(group["frame_id"].values,
                                group["x"].values,
                                group["y"].values))
    return peds


def make_static_plot(peds_by_id, out_path, scene_name):
    """Plot every pedestrian's full trajectory as a single image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for ped_id, track in peds_by_id.items():
        xs = [p[1] for p in track]
        ys = [p[2] for p in track]
        ax.plot(xs, ys, linewidth=0.8, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(f"All pedestrian tracks — {scene_name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved static tracks to {out_path}")


def animate_scene(df, frames_by_id, peds_by_id, scene_name, out_path):
    """Build and save the animation."""
    # All frame ids in order
    frame_ids = sorted(frames_by_id.keys())
    if FRAME_STRIDE > 1:
        frame_ids = frame_ids[::FRAME_STRIDE]
    if MAX_FRAMES is not None:
        frame_ids = frame_ids[:MAX_FRAMES]

    # Fixed axis limits across the whole scene
    x_min, x_max = df["x"].min() - MARGIN, df["x"].max() + MARGIN
    y_min, y_max = df["y"].min() - MARGIN, df["y"].max() + MARGIN

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Drawing artists we'll re-use each frame
    points_artist, = ax.plot([], [], "o", color="tab:blue", markersize=5)
    trail_lines = []   # list of Line2D objects, recreated each frame
    text_labels = []   # list of Text objects, recreated each frame
    title = ax.set_title("")

    def update(frame_id):
        # Clear previous trails and labels
        for line in trail_lines:
            line.remove()
        trail_lines.clear()
        for txt in text_labels:
            txt.remove()
        text_labels.clear()

        # Pedestrians present at this frame
        peds_here = frames_by_id.get(frame_id, [])
        xs = [p[1] for p in peds_here]
        ys = [p[2] for p in peds_here]
        points_artist.set_data(xs, ys)

        # Draw a short trail for each present pedestrian
        for ped_id, x, y in peds_here:
            track = peds_by_id[ped_id]
            # Keep only points up to and including the current frame
            recent = [(f, tx, ty) for (f, tx, ty) in track if f <= frame_id]
            if TRAIL_LENGTH is not None:
                recent = recent[-TRAIL_LENGTH:]
            if len(recent) > 1:
                tx = [r[1] for r in recent]
                ty = [r[2] for r in recent]
                line, = ax.plot(tx, ty, color="tab:blue",
                                linewidth=0.8, alpha=0.5)
                trail_lines.append(line)
            if SHOW_PED_IDS:
                label = ax.text(x, y, str(ped_id),
                                fontsize=6, color="black")
                text_labels.append(label)

        title.set_text(f"{scene_name}  |  frame {frame_id}  "
                       f"|  peds visible: {len(peds_here)}")
        return [points_artist, title, *trail_lines, *text_labels]

    anim = FuncAnimation(fig, update, frames=frame_ids,
                         interval=1000 / FPS, blit=False)

    # Pick a writer based on what's available. Try ffmpeg first for .mp4,
    # and fall back to a .gif via Pillow if it isn't available or fails.
    out_path = Path(out_path)
    total = len(frame_ids)
    saved = False

    if out_path.suffix == ".mp4":
        try:
            writer = FFMpegWriter(fps=FPS)
            with tqdm(total=total, desc="Animating", unit="frame") as bar:
                anim.save(out_path, writer=writer, dpi=120,
                          progress_callback=lambda i, n: bar.update(1))
            print(f"Saved animation to {out_path}")
            saved = True
        except Exception as err:
            print(f"ffmpeg writer failed ({err}). Falling back to .gif")

    if not saved:
        if out_path.suffix == ".mp4":
            out_path = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=FPS)
        with tqdm(total=total, desc="Animating", unit="frame") as bar:
            anim.save(out_path, writer=writer, dpi=100,
                      progress_callback=lambda i, n: bar.update(1))
        print(f"Saved animation to {out_path}")

    plt.close(fig)


def main():
    raw_path = Path(RAW_FILE)
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find raw file: {raw_path}")

    scene_name = raw_path.stem
    print(f"Loading {raw_path}")

    df = load_scene(raw_path)
    print(f"  rows: {len(df)}")
    print(f"  unique frames: {df['frame_id'].nunique()}")
    print(f"  unique pedestrians: {df['ped_id'].nunique()}")

    frames_by_id = group_by_frame(df)
    peds_by_id = group_by_ped(df)

    make_static_plot(peds_by_id, STATIC_OUT, scene_name)
    animate_scene(df, frames_by_id, peds_by_id, scene_name, ANIMATION_OUT)


if __name__ == "__main__":
    main()
