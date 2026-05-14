"""Self-contained navigation demo.

Loads the raw univ scene, picks a dense anchor frame, runs the SSM on each
pedestrian's 8-frame observed history, and rolls out an ego robot with a
potential-field controller that re-predicts every step.

Run from project root:   python -m navigation.simulate_simple
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ssm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import TrajectoryPredictor                    # noqa: E402
from controller import compute_velocity, at_goal         # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE_NAME = "univ"
RAW_FILE = PROJECT_ROOT / "datasets" / SCENE_NAME / "test" / "students003.txt"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "ssm" / SCENE_NAME / "v2"
    / "lr_0.003_batch_512_epochs_150_best_model.pt"
)

OBSERVE_LEN = 8       # frames of observed history fed to the model
PREDICT_LEN = 12      # frames of future the model produces
FRAME_STEP = 10       # raw video frames per trajectory step (dataset convention)
DT_SECONDS = 0.4      # seconds per trajectory step

# Controller gains
K_ATT = 1.0
K_REP = 2.5
INFLUENCE_RADIUS = 2.0
MAX_SPEED = 1.3
GOAL_TOLERANCE = 0.4

# Ego start/goal in scene coordinates (meters)
EGO_START = np.array([3.0, 1.0])
EGO_GOAL = np.array([13.0, 13.0])
MAX_SIM_STEPS = 80

# Pedestrian thinning
KEEP_N_PEDS = None    # None means keep every ped present in the rollout window
RANDOM_SEED = 0
ANCHOR_FRAME = None   # None means auto-pick the densest frame

# Output files
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "navigation_rollout_simple.npz"
METADATA_FILE = RESULTS_DIR / "navigation_rollout_simple_metadata.json"
OUTPUT_PLOT = RESULTS_DIR / "navigation_rollout_simple.png"


# ---------------------------------------------------------------------------
# Scene loading and indexing
# ---------------------------------------------------------------------------
def load_scene(path):
    """Read the raw tab-separated file into a DataFrame.

    Columns in the file are: frame_id, ped_id, x, y (all floats).
    """
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["frame_id", "ped_id", "x", "y"])
    return df


def index_pedestrians(df):
    """Group rows by pedestrian id.

    Returns {ped_id: {frame_id: np.array([x, y])}} for fast lookup.
    Both ped_id and frame_id are kept as Python ints.
    """
    ped_index = {}
    for _, row in df.iterrows():
        pid = int(row["ped_id"])
        frame = int(row["frame_id"])
        if pid not in ped_index:
            ped_index[pid] = {}
        ped_index[pid][frame] = np.array([row["x"], row["y"]], dtype=np.float32)
    return ped_index


def find_best_anchor(ped_index, frame_step, observe_len):
    """Pick the frame with the most peds that have a full observe_len history.

    We slide every candidate anchor frame in the data and count how many
    peds have all of {anchor - (observe_len-1)*step, ..., anchor} present
    in their frame map. Return (anchor, count).
    """
    all_frames = set()
    for fmap in ped_index.values():
        all_frames.update(fmap.keys())

    best_anchor = None
    best_count = -1
    for anchor in sorted(all_frames):
        count = 0
        for fmap in ped_index.values():
            ok = True
            for k in range(observe_len):
                if (anchor - k * frame_step) not in fmap:
                    ok = False
                    break
            if ok:
                count += 1
        if count > best_count:
            best_count = count
            best_anchor = anchor
    return best_anchor, best_count


def get_observed(ped_index, anchor, frame_step, observe_len):
    """Return {ped_id: array(observe_len, 2)} for peds with a full history.

    A ped is "fully observed" if all of frames anchor - (observe_len-1)*step
    through anchor are present.
    """
    observed = {}
    for pid, fmap in ped_index.items():
        history = []
        for k in range(observe_len):
            frame = anchor - (observe_len - 1 - k) * frame_step
            if frame not in fmap:
                history = None
                break
            history.append(fmap[frame])
        if history is not None:
            observed[pid] = np.stack(history, axis=0)  # (observe_len, 2)
    return observed


def get_future_truth(ped_index, anchor, frame_step, predict_len):
    """Return {ped_id: array(<=predict_len, 2)} of true positions after the anchor.

    Stops early if a ped exits the scene before predict_len frames.
    Only includes peds that have at least one future frame available.
    """
    future = {}
    for pid, fmap in ped_index.items():
        positions = []
        for k in range(1, predict_len + 1):
            frame = anchor + k * frame_step
            if frame not in fmap:
                break
            positions.append(fmap[frame])
        if positions:
            future[pid] = np.stack(positions, axis=0)
    return future


# ---------------------------------------------------------------------------
# SSM prediction
# ---------------------------------------------------------------------------
def predict_pedestrians(model, observed, device):
    """Run the SSM on all observed peds and return absolute future positions.

    The model expects step-to-step displacements as input and produces
    step-to-step displacements as output for indices observe_len..19. We
    reconstruct absolute positions by cumulatively summing the predicted
    deltas onto the last observed position.

    observed: {ped_id: array(observe_len, 2)}
    returns:  {ped_id: array(predict_len, 2)} of absolute predicted positions
    """
    if not observed:
        return {}

    ped_ids = list(observed.keys())
    histories = np.stack([observed[pid] for pid in ped_ids], axis=0)  # (P, 8, 2)
    histories_t = torch.from_numpy(histories).float().to(device)

    batch_size = histories_t.shape[0]
    full_seq_len = OBSERVE_LEN + PREDICT_LEN  # 20

    # Build model input: deltas at indices 1..OBSERVE_LEN-1, zeros elsewhere.
    model_input = torch.zeros(batch_size, full_seq_len, 2, device=device)
    model_input[:, 1:OBSERVE_LEN, :] = (
        histories_t[:, 1:OBSERVE_LEN, :] - histories_t[:, :OBSERVE_LEN - 1, :]
    )

    model.eval()
    with torch.no_grad():
        raw_output = model(model_input)  # (P, 20, 2)

    # Convert predicted deltas back to absolute positions.
    last_pos = histories_t[:, OBSERVE_LEN - 1:OBSERVE_LEN, :]  # (P, 1, 2)
    predicted_deltas = raw_output[:, OBSERVE_LEN:, :]          # (P, 12, 2)
    predicted_abs = last_pos + predicted_deltas.cumsum(dim=1)  # (P, 12, 2)
    predicted_abs = predicted_abs.cpu().numpy()

    return {pid: predicted_abs[i] for i, pid in enumerate(ped_ids)}


# ---------------------------------------------------------------------------
# Pedestrian thinning and display arrays
# ---------------------------------------------------------------------------
def subsample_pedestrians(ped_index, n, seed, rollout_frames):
    """Randomly keep at most n pedestrians that appear in the rollout window.

    Drop anyone never present during the simulation horizon first, then
    sample n from what's left. This makes KEEP_N_PEDS a meaningful budget
    instead of burning slots on peds from a different part of the recording.
    """
    eligible = [pid for pid, fmap in ped_index.items()
                if not rollout_frames.isdisjoint(fmap.keys())]
    if n is None or n >= len(eligible):
        return {pid: ped_index[pid] for pid in eligible}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    return {pid: ped_index[pid] for pid in chosen}


def build_pedestrian_positions(ped_index, ped_ids, anchor, frame_step, num_steps):
    """Build (T+1, P, 2) positions and (T+1, P) mask for animation.

    At each sim step t (0..num_steps), look up frame anchor + t*frame_step
    for each ped. Missing frames get NaN position and False mask.
    """
    num_frames = num_steps + 1
    num_peds = len(ped_ids)
    positions = np.full((num_frames, num_peds, 2), np.nan, dtype=np.float32)
    mask = np.zeros((num_frames, num_peds), dtype=bool)

    for t in range(num_frames):
        frame = anchor + t * frame_step
        for j, pid in enumerate(ped_ids):
            if frame in ped_index[pid]:
                positions[t, j] = ped_index[pid][frame]
                mask[t, j] = True
    return positions, mask


# ---------------------------------------------------------------------------
# Ego rollout
# ---------------------------------------------------------------------------
def simulate_ego(start, goal, model, device, ped_index, anchor):
    """Roll out the ego, re-predicting every step.

    At each sim step k, get every ped with a full 8-frame history ending
    at the current frame and use the model's next-step prediction as the
    obstacle position. No active-set, no refill — just whoever is eligible.
    """
    ego_pos = start.astype(np.float64).copy()
    positions = [ego_pos.copy()]
    velocities = []
    obstacle_per_step = []
    reached = False

    for step in range(MAX_SIM_STEPS):
        if at_goal(ego_pos, goal, GOAL_TOLERANCE):
            reached = True
            break

        current_frame = anchor + step * FRAME_STEP
        observed_now = get_observed(ped_index, current_frame, FRAME_STEP, OBSERVE_LEN)
        predicted_now = predict_pedestrians(model, observed_now, device)

        step_obstacles = {pid: pred[0] for pid, pred in predicted_now.items()}
        obstacle_per_step.append(step_obstacles)

        obstacles = list(step_obstacles.values())
        velocity = compute_velocity(
            ego_pos, goal, obstacles,
            K_ATT, K_REP, INFLUENCE_RADIUS, MAX_SPEED,
        )
        velocities.append(velocity.copy())
        ego_pos = ego_pos + velocity * DT_SECONDS
        positions.append(ego_pos.copy())

    # Build the union of pedestrian ids seen across all steps.
    union_ids = []
    id_to_col = {}
    for step_obs in obstacle_per_step:
        for pid in step_obs:
            if pid not in id_to_col:
                id_to_col[pid] = len(union_ids)
                union_ids.append(pid)

    num_obs_peds = len(union_ids)
    num_steps = len(velocities)
    obstacle_positions = np.full((num_steps, num_obs_peds, 2), np.nan, dtype=np.float32)
    obstacle_mask = np.zeros((num_steps, num_obs_peds), dtype=bool)
    for t, step_obs in enumerate(obstacle_per_step):
        for pid, pos in step_obs.items():
            col = id_to_col[pid]
            obstacle_positions[t, col] = pos
            obstacle_mask[t, col] = True

    if velocities:
        ego_velocities = np.array(velocities, dtype=np.float32)
    else:
        ego_velocities = np.zeros((0, 2), dtype=np.float32)

    return {
        "ego_positions":      np.array(positions, dtype=np.float32),
        "ego_velocities":     ego_velocities,
        "obstacle_positions": obstacle_positions,
        "obstacle_mask":      obstacle_mask,
        "obstacle_ped_ids":   union_ids,
        "reached_goal":       reached,
    }


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------
def save_rollout(rollout, observed, predicted, ped_positions, ped_mask, anchor,
                 results_file, metadata_file):
    """Save the rollout arrays as npz and metadata as json."""
    results_file.parent.mkdir(parents=True, exist_ok=True)

    anchor_ids = list(observed.keys())
    if anchor_ids:
        observed_arr = np.stack([observed[pid] for pid in anchor_ids], axis=0)
        predicted_arr = np.stack([predicted[pid] for pid in anchor_ids], axis=0)
    else:
        observed_arr = np.zeros((0, OBSERVE_LEN, 2), dtype=np.float32)
        predicted_arr = np.zeros((0, PREDICT_LEN, 2), dtype=np.float32)

    np.savez(
        results_file,
        ego_positions=rollout["ego_positions"],
        ego_velocities=rollout["ego_velocities"],
        start=EGO_START.astype(np.float32),
        goal=EGO_GOAL.astype(np.float32),
        pedestrian_positions=ped_positions,
        pedestrian_mask=ped_mask,
        pedestrian_observed=observed_arr.astype(np.float32),
        predicted_positions=predicted_arr.astype(np.float32),
        obstacle_positions=rollout["obstacle_positions"],
        obstacle_mask=rollout["obstacle_mask"],
    )

    metadata = {
        "scene_name": SCENE_NAME,
        "raw_scene_file": str(RAW_FILE),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "observe_len": OBSERVE_LEN,
        "predict_len": PREDICT_LEN,
        "prediction_horizon": PREDICT_LEN,
        "frame_step": FRAME_STEP,
        "dt_seconds": DT_SECONDS,
        "max_speed": MAX_SPEED,
        "k_att": K_ATT,
        "k_rep": K_REP,
        "influence_radius": INFLUENCE_RADIUS,
        "goal_tolerance": GOAL_TOLERANCE,
        "max_sim_steps": MAX_SIM_STEPS,
        "anchor_frame": anchor,
        "num_steps_done": int(rollout["ego_velocities"].shape[0]),
        "num_pedestrians": len(anchor_ids),
        "num_obstacle_pedestrians": len(rollout["obstacle_ped_ids"]),
        "re_predict_every_step": True,
        "reached_goal": bool(rollout["reached_goal"]),
        "start": EGO_START.tolist(),
        "goal": EGO_GOAL.tolist(),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved rollout  -> {results_file}")
    print(f"Saved metadata -> {metadata_file}")


def plot_result(observed, predicted, future_truth, ego_path, anchor, output_plot):
    """Static plot: observed history, predicted future, true future, ego path."""
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, pid in enumerate(observed):
        hist = observed[pid]
        ax.plot(hist[:, 0], hist[:, 1], "-", color="0.5", linewidth=1.0,
                alpha=0.6, label="Observed history" if i == 0 else None)

    for i, pid in enumerate(predicted):
        pred = predicted[pid]
        ax.plot(pred[:, 0], pred[:, 1], "r--", linewidth=1.0, alpha=0.7,
                label="Predicted future" if i == 0 else None)

    drew_truth_label = False
    for pid, truth in future_truth.items():
        label = "True future" if not drew_truth_label else None
        ax.plot(truth[:, 0], truth[:, 1], "g-", linewidth=1.0, alpha=0.7,
                label=label)
        drew_truth_label = True

    ax.plot(ego_path[:, 0], ego_path[:, 1], "b-", linewidth=2.0, label="Ego path")
    ax.plot(EGO_START[0], EGO_START[1], "bs", markersize=10, label="Start")
    ax.plot(EGO_GOAL[0], EGO_GOAL[1], "b*", markersize=16, label="Goal")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Navigation rollout - {SCENE_NAME} (anchor frame {anchor})")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=120)
    plt.close(fig)
    print(f"Saved static plot -> {output_plot}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: No trained checkpoint at {CHECKPOINT_PATH}")
        print("Train the SSM model first:  cd ssm && python train.py")
        sys.exit(1)

    df = load_scene(RAW_FILE)
    ped_index = index_pedestrians(df)
    print(f"Loaded {len(ped_index)} pedestrians from {RAW_FILE.name}")

    # Pick the anchor first from the FULL pool, so anchor selection sees
    # every candidate frame.
    if ANCHOR_FRAME is None:
        anchor, count = find_best_anchor(ped_index, FRAME_STEP, OBSERVE_LEN)
        print(f"Auto-selected anchor frame {anchor} with {count} fully-observed peds")
    else:
        anchor = ANCHOR_FRAME
        print(f"Using configured anchor frame {anchor}")

    # Now thin the pool. Only consider peds that are actually present
    # somewhere in the rollout window — sampling from the full recording
    # wastes most of the budget on peds from a different time.
    rollout_frames = {anchor + t * FRAME_STEP for t in range(MAX_SIM_STEPS + 1)}
    if KEEP_N_PEDS is not None:
        ped_index = subsample_pedestrians(ped_index, KEEP_N_PEDS, RANDOM_SEED, rollout_frames)
        print(f"Randomly kept {len(ped_index)} pedestrians present in rollout window (seed={RANDOM_SEED})")

    observed = get_observed(ped_index, anchor, FRAME_STEP, OBSERVE_LEN)
    future_truth = get_future_truth(ped_index, anchor, FRAME_STEP, PREDICT_LEN)
    print(f"Pedestrians with full 8-frame history at anchor: {len(observed)}")

    if not observed:
        print("No pedestrians available at this anchor. Try a different ANCHOR_FRAME.")
        sys.exit(1)

    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH.name}")

    predicted = predict_pedestrians(model, observed, device)

    rollout = simulate_ego(EGO_START, EGO_GOAL, model, device, ped_index, anchor)
    ego_path = rollout["ego_positions"]
    num_steps_done = rollout["ego_velocities"].shape[0]
    final_dist = float(np.linalg.norm(ego_path[-1] - EGO_GOAL))
    print(
        f"Ego rollout: {ego_path.shape[0]} positions ({num_steps_done} steps), "
        f"final distance to goal {final_dist:.2f} m, "
        f"reached={rollout['reached_goal']}"
    )
    print(
        f"Obstacles seen across the rollout: "
        f"{len(rollout['obstacle_ped_ids'])} unique pedestrians"
    )

    # Decoupled display: include every ped from the (thinned) pool who shows
    # up in any rollout frame, not just the anchor cohort. Peds appear as
    # black dots whenever they're in the raw data, with no spawn artifact.
    num_rollout_steps = ego_path.shape[0] - 1
    rollout_display_frames = {anchor + t * FRAME_STEP for t in range(num_rollout_steps + 1)}
    display_ped_ids = sorted([
        pid for pid, fmap in ped_index.items()
        if not rollout_display_frames.isdisjoint(fmap.keys())
    ])
    print(f"Display peds (any appearance in rollout window): {len(display_ped_ids)}")
    ped_positions, ped_mask = build_pedestrian_positions(
        ped_index, display_ped_ids, anchor, FRAME_STEP, num_rollout_steps,
    )

    save_rollout(rollout, observed, predicted, ped_positions, ped_mask, anchor,
                 RESULTS_FILE, METADATA_FILE)
    plot_result(observed, predicted, future_truth, ego_path, anchor, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
