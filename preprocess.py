from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Parameters ──────────────────────────────────────────────────────────────
# Path to the leave-one-out fold we're using (held-out scene = eth)
scene_name = "eth"
DATA_ROOT = Path.cwd() / "datasets" / scene_name

# Where to write the processed tensors
OUTPUT_DIR = Path.cwd() / "datasets_processed" / scene_name

OBSERVE_LEN = 8   # number of frames we observe (3.2 s at 0.4 s/frame)
PREDICT_LEN = 12  # number of frames we predict (4.8 s)
SEQ_LEN = OBSERVE_LEN + PREDICT_LEN  # 20 total frames per trajectory window
FRAME_STEP = 10  # all files in this dataset annotate every 10th frame


def read_file(filepath, file_index):
    """Read one .txt file and return a DataFrame with columns
    [frame_id, ped_id, x, y].

    The raw files have exactly 4 tab-separated columns: frame_id, ped_id, x, y.
    We make each ped_id unique by appending the file index, so ped #1 from
    file A is not confused with ped #1 from file B.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["frame_id", "ped_id", "x", "y"],
    )

    # Tag each pedestrian ID with the file it came from
    df["ped_id"] = df["ped_id"].astype(str) + "_f" + str(file_index)

    return df



def extract_trajectories_from_df(df, frame_step):
    """Slide a 20-frame window over each pedestrian's path.

    A window is kept only if the pedestrian appears in ALL 20 consecutive
    frames with no gaps (every step must equal frame_step exactly).

    Returns a list of numpy arrays, each with shape (20, 2).
    """
    trajectories = []

    for ped_id, ped_data in df.groupby("ped_id"):
        # Sort this person's rows by time
        ped_data = ped_data.sort_values("frame_id").reset_index(drop=True)

        frame_ids = ped_data["frame_id"].values
        xs = ped_data["x"].values
        ys = ped_data["y"].values

        # Try every possible 20-frame starting position for this person
        for start in range(len(frame_ids) - SEQ_LEN + 1):
            window_frames = frame_ids[start : start + SEQ_LEN]

            # Check that every consecutive pair differs by exactly frame_step
            diffs = window_frames[1:] - window_frames[:-1]
            if not np.all(diffs == frame_step):
                continue  # gap in trajectory — skip this window

            window_x = xs[start : start + SEQ_LEN]
            window_y = ys[start : start + SEQ_LEN]

            # One row per timestep → shape (20, 2)
            trajectory = np.stack([window_x, window_y], axis=1)
            trajectories.append(trajectory)

    return trajectories


def normalize_trajectories(trajectories):
    """Shift each trajectory so the position at timestep 7 becomes the origin.

    Timestep 7 is the last observed frame (index 7 in 0-indexed 20 frames).
    After this, all 20 positions are expressed relative to where the agent
    stood when we stopped watching them.
    """
    normalized = []
    for trajectory in trajectories:
        anchor = trajectory[OBSERVE_LEN - 1]  # [x, y] at last observed frame
        trajectory_centered = trajectory - anchor  # subtract from all 20 rows
        normalized.append(trajectory_centered)
    return normalized


def process_split(split_folder):
    """Run the full pipeline for one split and return a float32 tensor.

    Each file is processed independently (its own frame step), and trajectories
    are collected and merged at the end.
    """
    print(f"\nProcessing split: {split_folder}")

    txt_files = sorted(p.name for p in Path(split_folder).iterdir() if p.suffix == ".txt")

    all_trajectories = []

    for file_index, filename in enumerate(tqdm(txt_files, desc="  Files")):
        filepath = Path(split_folder) / filename

        df = read_file(filepath, file_index)
        file_trajectories = extract_trajectories_from_df(df, FRAME_STEP)
        all_trajectories.extend(file_trajectories)

    if len(all_trajectories) == 0:
        raise ValueError(f"No trajectories found in {split_folder} — check data path.")

    normalized = normalize_trajectories(all_trajectories)

    # np.array stacks the list of (20, 2) arrays into one (N, 20, 2) array
    tensor_data = torch.tensor(np.array(normalized), dtype=torch.float32)
    print(f"\n  Total trajectories: {len(all_trajectories)}")
    print(f"  Final tensor shape: {tensor_data.shape}  (N x 20 steps x 2 coords)")

    return tensor_data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_folder = DATA_ROOT / split_name
        tensor_data = process_split(split_folder)

        output_path = OUTPUT_DIR / f"{split_name}.pt"
        torch.save(tensor_data, output_path)
        print(f"  Saved to {output_path}")

    print("\nAll splits done!")


if __name__ == "__main__":
    main()


# =============================================================================
# DATASET STRUCTURE AND PREPROCESSING OVERVIEW
# =============================================================================
#
# RAW DATASET LAYOUT
# ------------------
# The datasets/ folder contains pedestrian trajectory data from five real-world
# scenes recorded by the ETH and UCY research groups:
#
#   datasets/
#   ├── eth/       -- ETH pedestrian area (Zurich)
#   ├── hotel/     -- ETH hotel area (Zurich)
#   ├── univ/      -- UCY university campus
#   ├── zara1/     -- UCY Zara shopping street (sequence 1)
#   ├── zara2/     -- UCY Zara shopping street (sequence 2)
#   └── raw/       -- original unprocessed files (not used here)
#
# Each scene folder has three subfolders: train/, val/, test/.
# This is the Social-GAN "leave-one-out" split format: to evaluate on one
# scene, that scene's data goes in test/, and the other four scenes' data
# is split across train/ and val/.
#
# For example, in datasets/eth/:
#   test/   contains biwi_eth.txt              <- the ETH scene we evaluate on
#   train/  contains hotel, zara1/2, univ data <- earlier segments of the other 4 scenes
#   val/    contains the same four scenes      <- later segments of the same recordings
#
# train/ and val/ cover the same four scenes, not different ones. The split is
# temporal: Social-GAN cuts each scene's video chronologically, putting the
# earlier portion in train/ and the later portion in val/.
#
# FILE FORMAT
# -----------
# Every .txt file is tab-separated with exactly 4 columns and no header:
#
#   frame_id    ped_id    x    y
#
#   frame_id : integer frame number from the original video recording.
#              Multiple pedestrians share the same frame_id when they appear
#              in the same video frame.
#   ped_id   : integer ID for each tracked pedestrian. IDs are only unique
#              within a single file — the same number in two different files
#              refers to two different people.
#   x, y     : world-space position in meters, already converted from pixels.
#
# Frame rate note: all files in this dataset use a frame step of 10
# (frame_ids jump by 10: 0, 10, 20, ...), which corresponds to a 0.4-second
# real-time interval at 25 fps.
#
# WHAT THIS SCRIPT DOES (STEP BY STEP)
# -------------------------------------
# 1. READ
#    For each split (train, val, test), we walk the corresponding folder and
#    read every .txt file into a pandas DataFrame with columns
#    [frame_id, ped_id, x, y]. To avoid ID collisions across files, we tag
#    each pedestrian as "ped_id_fN" where N is the file's index in the folder.
#
# 2. EXTRACT TRAJECTORY WINDOWS
#    For each pedestrian in a file, we slide a 20-frame window over their
#    recorded positions. A window is kept only if the pedestrian appears in
#    all 20 frames with no gaps — every consecutive frame pair must differ
#    by exactly 10. The first 8 frames are the "observed"
#    portion; the last 12 are the "future" portion the model must predict.
#    Pedestrians are processed independently (no social interactions yet).
#
# 3. NORMALIZE
#    Each 20-frame trajectory is shifted so that the position at timestep 7
#    (the last observed frame) becomes the origin (0, 0). All 20 positions
#    are expressed relative to that anchor. This removes absolute location
#    bias and lets the model focus on relative motion patterns.
#
# 4. SAVE
#    Trajectories are stacked into a tensor of shape (N, 20, 2) — N windows,
#    20 timesteps each, 2 coordinates (x, y) — and saved as a PyTorch .pt
#    file in datasets_processed/{scene_name}/ for each split.
#
# OUTPUT
# ------
#   datasets_processed/eth/train.pt   shape ~ (30000, 20, 2)
#   datasets_processed/eth/val.pt     shape ~ (5000,  20, 2)
#   datasets_processed/eth/test.pt    shape ~ (364,   20, 2)
# =============================================================================
