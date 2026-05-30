"""Navigation demo: load the univ scene, predict pedestrian futures with the
SSM, and roll out an ego robot using the potential field controller.

Run from project root:  python -m navigation.simulate_simple
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ssm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import TrajectoryPredictor
from controller import compute_velocity, at_goal, constant_velocity_predict

SCENE_NAME = "univ"
RAW_FILE = PROJECT_ROOT / "datasets" / SCENE_NAME / "test" / "students003.txt"
CHECKPOINT_PATH = (PROJECT_ROOT / "ssm" / SCENE_NAME / "v2" / "lr_0.003_batch_512_epochs_50_best_model.pt")

OBSERVE_LEN = 8      # observed frames fed to the model
PREDICT_LEN = 12     # frames the model predicts
FRAME_STEP = 10      # raw frames per trajectory step
DT_SECONDS = 0.4     # seconds per step

# controller gains
K_ATT = 1.0  # pull toward goal
K_REP = 2.5  # push from obstacles
INFLUENCE_RADIUS = 2.0  # meters
MAX_SPEED = 1.3  # m/s
GOAL_TOLERANCE = 0.4  # meters

# how many predicted future steps the controller reacts to, and their discount
PREDICT_HORIZON = 8
GAMMA = 0.8 # per-step discount in (0, 1]

EGO_START = np.array([0.0, 0.0])
EGO_GOAL = np.array([15.0, 15.0])
MAX_SIM_STEPS = 200

# pedestrian thinning
KEEP_PED_FRACTION = 0.50  # fraction of peds present in the rollout window to keep
RANDOM_SEED = 0
ANCHOR_FRAME = 1040  # densest frame in students003.txt (46 fully-observed peds)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "navigation_rollout_simple.npz"
METADATA_FILE = RESULTS_DIR / "navigation_rollout_simple_metadata.json"


def index_pedestrians(df):
    """Group raw rows by pedestrian id.

    input:
        df: DataFrame from load_scene
    output:
        {ped_id: {frame_id: np.array([x, y])}}
    """
    ped_index = {}
    for pid, group in df.groupby("ped_id"):
        pid = int(pid)
        frames = {}
        for _, row in group.iterrows():
            frame = int(row["frame_id"])
            pos = np.array([row["x"], row["y"]], dtype=np.float32)
            frames[frame] = pos
        ped_index[pid] = frames
    return ped_index


def get_observed(ped_index, anchor, frame_step, observe_len):
    """Collect peds with a full history ending at the anchor.

    input:
        ped_index: {ped_id: {frame_id: pos}}
        anchor: anchor frame id
        frame_step: raw frames between trajectory steps
        observe_len: required history length
    output:
        {ped_id: array(observe_len, 2)} in chronological order
    """
    observed = {}
    for pid, fmap in ped_index.items():
        history = []
        complete = True
        for k in range(observe_len):
            frame = anchor - (observe_len - 1 - k) * frame_step
            if frame not in fmap:
                complete = False
                break
            history.append(fmap[frame])
        if complete:
            observed[pid] = np.stack(history, axis=0)
    return observed


def predict_pedestrians(model, observed, device):
    """Run the SSM on all observed peds.

    The model takes step deltas and outputs step deltas. We cumsum the
    predicted deltas onto the last observed position to get absolute coords.

    input:
        model: trained TrajectoryPredictor
        observed: {ped_id: array(OBSERVE_LEN, 2)}
        device: torch device
    output:
        {ped_id: array(PREDICT_LEN, 2)} absolute predicted positions
    """
    if not observed:
        return {}

    ped_ids = list(observed.keys())
    history_list = []
    for pid in ped_ids:
        history_list.append(observed[pid])
    histories = np.stack(history_list, axis=0)
    histories_t = torch.from_numpy(histories).float().to(device)

    batch_size = histories_t.shape[0]
    full_seq_len = OBSERVE_LEN + PREDICT_LEN

    # input deltas at indices 1..OBSERVE_LEN-1, zeros elsewhere
    model_input = torch.zeros(batch_size, full_seq_len, 2, device=device)
    model_input[:, 1:OBSERVE_LEN, :] = (
        histories_t[:, 1:OBSERVE_LEN, :] - histories_t[:, :OBSERVE_LEN - 1, :]
    )

    model.eval()
    with torch.no_grad():
        raw_output = model(model_input)

    last_pos = histories_t[:, OBSERVE_LEN - 1:OBSERVE_LEN, :]
    predicted_deltas = raw_output[:, OBSERVE_LEN:, :]
    predicted_abs = last_pos + predicted_deltas.cumsum(dim=1)
    predicted_abs = predicted_abs.cpu().numpy()

    result = {}
    for i in range(len(ped_ids)):
        pid = ped_ids[i]
        result[pid] = predicted_abs[i]
    return result


def get_partial_history(ped_index, pid, current_frame, frame_step, observe_len):
    """Walk back from current_frame collecting consecutive samples.

    input:
        ped_index: {ped_id: {frame_id: pos}}
        pid: pedestrian id
        current_frame: frame to walk back from
        frame_step: raw frames between trajectory steps
        observe_len: max samples to gather
    output:
        array(k, 2) in chronological order with k <= observe_len, or None if
        the ped is not present at current_frame
    """
    fmap = ped_index[pid]
    if current_frame not in fmap:
        return None
    samples = [fmap[current_frame]]
    for k in range(1, observe_len):
        frame = current_frame - k * frame_step
        if frame not in fmap:
            break
        samples.append(fmap[frame])
    samples.reverse()
    return np.stack(samples, axis=0)


def subsample_pedestrians(ped_index, fraction, seed, rollout_frames):
    """Randomly thin the pedestrian pool.

    input:
        ped_index: {ped_id: {frame_id: pos}}
        fraction: keep ratio in (0, 1)
        seed: rng seed
        rollout_frames: set of frame ids covered by the simulation
    output:
        thinned {ped_id: {frame_id: pos}} restricted to eligible peds
    """
    # keep only peds that appear in at least one frame of the rollout window
    eligible = []
    for pid, fmap in ped_index.items():
        present_in_window = False
        for frame in fmap:
            if frame in rollout_frames:
                present_in_window = True
                break
        if present_in_window:
            eligible.append(pid)

    n = int(round(fraction * len(eligible)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    result = {}
    for pid in chosen:
        result[pid] = ped_index[pid]
    return result


def build_pedestrian_positions(ped_index, ped_ids, anchor, frame_step, num_steps):
    """Lay out ped positions per sim step for the animation.

    input:
        ped_index: {ped_id: {frame_id: pos}}
        ped_ids: ordered list of pedestrian ids to include
        anchor: anchor frame id (step 0)
        frame_step: raw frames between trajectory steps
        num_steps: number of sim steps after the anchor
    output:
        (positions, mask): positions is (T+1, P, 2) with NaN where missing,
        mask is (T+1, P) bool
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


def simulate_ego(start, goal, model, device, ped_index, anchor):
    """Roll the ego forward, re-predicting pedestrian futures every step.

    input:
        start: (2,) ego start position
        goal: (2,) ego goal position
        model: trained TrajectoryPredictor
        device: torch device
        ped_index: {ped_id: {frame_id: pos}}
        anchor: anchor frame id (step 0)
    output:
        dict with ego_positions, ego_velocities, obstacle_ped_ids,
        step_predicted_positions/mask, and reached_goal
    """
    ego_pos = start.astype(np.float64).copy()
    positions = [ego_pos.copy()]
    velocities = []
    predictions_per_step = []
    reached = False

    for step in range(MAX_SIM_STEPS):
        if at_goal(ego_pos, goal, GOAL_TOLERANCE):
            reached = True
            break

        current_frame = anchor + step * FRAME_STEP
        observed_now = get_observed(ped_index, current_frame, FRAME_STEP, OBSERVE_LEN)
        predicted_now = predict_pedestrians(model, observed_now, device)

        # fallback: const-velocity for peds without a full 8-frame history
        for pid in ped_index:
            if pid in predicted_now:
                continue
            history = get_partial_history(ped_index, pid, current_frame,
                                          FRAME_STEP, OBSERVE_LEN)
            if history is None:
                continue
            predicted_now[pid] = constant_velocity_predict(history, PREDICT_HORIZON)

        # keep only the horizon the controller reacts to
        step_predictions = {}
        for pid in predicted_now:
            step_predictions[pid] = predicted_now[pid][:PREDICT_HORIZON]
        predictions_per_step.append(step_predictions)

        # stack to (P, H, 2) for the controller
        if step_predictions:
            prediction_list = []
            for pid in step_predictions:
                prediction_list.append(step_predictions[pid])
            horizon_array = np.stack(prediction_list, axis=0).astype(np.float64)
        else:
            horizon_array = np.zeros((0, PREDICT_HORIZON, 2), dtype=np.float64)

        velocity = compute_velocity(
            ego_pos, goal, horizon_array,
            K_ATT, K_REP, INFLUENCE_RADIUS, MAX_SPEED, GAMMA,
        )
        velocities.append(velocity.copy())
        ego_pos = ego_pos + velocity * DT_SECONDS
        positions.append(ego_pos.copy())

    # union of ped ids seen across all steps, mapped to column indices
    union_ids = []
    id_to_col = {}
    for step_preds in predictions_per_step:
        for pid in step_preds:
            if pid not in id_to_col:
                id_to_col[pid] = len(union_ids)
                union_ids.append(pid)

    num_obs_peds = len(union_ids)
    num_steps = len(velocities)
    step_predicted_positions = np.full(
        (num_steps, num_obs_peds, PREDICT_HORIZON, 2), np.nan, dtype=np.float32,
    )
    step_predicted_mask = np.zeros((num_steps, num_obs_peds), dtype=bool)
    for t in range(num_steps):
        step_preds = predictions_per_step[t]
        for pid in step_preds:
            col = id_to_col[pid]
            step_predicted_positions[t, col] = step_preds[pid]
            step_predicted_mask[t, col] = True

    if velocities:
        ego_velocities = np.array(velocities, dtype=np.float32)
    else:
        ego_velocities = np.zeros((0, 2), dtype=np.float32)

    return {
        "ego_positions": np.array(positions, dtype=np.float32),
        "ego_velocities": ego_velocities,
        "obstacle_ped_ids": union_ids,
        "step_predicted_positions": step_predicted_positions,
        "step_predicted_mask": step_predicted_mask,
        "reached_goal": reached,
    }


def save_rollout(rollout, ped_positions, ped_mask, anchor,
                 results_file, metadata_file):
    """Save rollout arrays as npz and run metadata as json.

    input:
        rollout: dict returned by simulate_ego
        ped_positions, ped_mask: arrays from build_pedestrian_positions
        anchor: anchor frame id
        results_file: output npz path
        metadata_file: output json path
    output:
        None (writes files to disk)
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_file,
        ego_positions=rollout["ego_positions"],
        start=EGO_START.astype(np.float32),
        goal=EGO_GOAL.astype(np.float32),
        pedestrian_positions=ped_positions,
        pedestrian_mask=ped_mask,
        step_predicted_positions=rollout["step_predicted_positions"],
        step_predicted_mask=rollout["step_predicted_mask"],
    )

    metadata = {
        "scene_name": SCENE_NAME,
        "raw_scene_file": str(RAW_FILE),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "observe_len": OBSERVE_LEN,
        "predict_len": PREDICT_LEN,
        "frame_step": FRAME_STEP,
        "dt_seconds": DT_SECONDS,
        "max_speed": MAX_SPEED,
        "k_att": K_ATT,
        "k_rep": K_REP,
        "predict_horizon": PREDICT_HORIZON,
        "gamma": GAMMA,
        "influence_radius": INFLUENCE_RADIUS,
        "goal_tolerance": GOAL_TOLERANCE,
        "max_sim_steps": MAX_SIM_STEPS,
        "anchor_frame": anchor,
        "num_steps_done": int(rollout["ego_velocities"].shape[0]),
        "num_obstacle_pedestrians": len(rollout["obstacle_ped_ids"]),
        "reached_goal": bool(rollout["reached_goal"]),
        "start": EGO_START.tolist(),
        "goal": EGO_GOAL.tolist(),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved rollout  -> {results_file}")
    print(f"Saved metadata -> {metadata_file}")


def main():
    """Run the full pipeline end to end.

    input:
        None (reads config constants at module top)
    output:
        None (writes npz and json to navigation/results/)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: No trained checkpoint at {CHECKPOINT_PATH}")
        print("Train the SSM model first:  cd ssm && python train.py")
        sys.exit(1)

    df = pd.read_csv(RAW_FILE, sep="\t", header=None,
                     names=["frame_id", "ped_id", "x", "y"])
    ped_index = index_pedestrians(df)
    print(f"Loaded {len(ped_index)} pedestrians from {RAW_FILE.name}")

    anchor = ANCHOR_FRAME
    print(f"Using anchor frame {anchor}")

    # then thin to peds that actually show up in the rollout window
    rollout_frames = set()
    for t in range(MAX_SIM_STEPS + 1):
        rollout_frames.add(anchor + t * FRAME_STEP)
    ped_index = subsample_pedestrians(ped_index, KEEP_PED_FRACTION, RANDOM_SEED, rollout_frames)
    print(f"Randomly kept {len(ped_index)} pedestrians "
          f"({KEEP_PED_FRACTION:.0%} of those present in rollout window, seed={RANDOM_SEED})")

    observed = get_observed(ped_index, anchor, FRAME_STEP, OBSERVE_LEN)
    print(f"Pedestrians with full 8-frame history at anchor: {len(observed)}")

    if not observed:
        print("No pedestrians available at this anchor. Try a different ANCHOR_FRAME.")
        sys.exit(1)

    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH.name}")

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

    # show every ped that appears anywhere in the rollout window, not just the anchor cohort
    num_rollout_steps = ego_path.shape[0] - 1
    rollout_display_frames = set()
    for t in range(num_rollout_steps + 1):
        rollout_display_frames.add(anchor + t * FRAME_STEP)

    display_ped_ids = []
    for pid, fmap in ped_index.items():
        appears_in_window = False
        for frame in fmap:
            if frame in rollout_display_frames:
                appears_in_window = True
                break
        if appears_in_window:
            display_ped_ids.append(pid)
    display_ped_ids = sorted(display_ped_ids)
    print(f"Display peds (any appearance in rollout window): {len(display_ped_ids)}")
    ped_positions, ped_mask = build_pedestrian_positions(
        ped_index, display_ped_ids, anchor, FRAME_STEP, num_rollout_steps,
    )

    save_rollout(rollout, ped_positions, ped_mask, anchor,
                 RESULTS_FILE, METADATA_FILE)


if __name__ == "__main__":
    main()
