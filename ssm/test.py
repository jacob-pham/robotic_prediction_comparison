from pathlib import Path
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from model import TrajectoryPredictor

# Paths and constants 
BATCH_SIZE = 512
LEARNING_RATE = 0.003
NUM_EPOCHS = 150

SCENE = "zara2" # scene to make test set, available scenes: "eth", "hotel", "univ", "zara1", "zara2"
VERSION = "v2"  # model version, refer to git commit history  

PROCESSED_DIR = Path.cwd().parent / "datasets_processed" / SCENE
CHECKPOINT_PATH = Path.cwd() / SCENE / VERSION / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_best_model.pt"
PREDICTIONS_PATH = Path.cwd() / SCENE / VERSION / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_predictions.png"

OBSERVE_LEN = 8    # frames we observe (indices 0–7)
PREDICT_LEN = 12   # frames we predict (indices 8–19)

NUM_PLOT = 9    # how many example trajectories to plot


def build_model_input(trajectory_tensor):
    """Turn positions into the step-delta model input (same as train.py).

    input:
        trajectory_tensor: (N, 20, 2) ground-truth positions
    output:
        (N, 20, 2) with observed deltas at indices 1..OBSERVE_LEN-1 and
        zeros elsewhere
    """
    model_input = torch.zeros_like(trajectory_tensor)
    model_input[:, 1:OBSERVE_LEN, :] = (
        trajectory_tensor[:, 1:OBSERVE_LEN, :] - trajectory_tensor[:, :OBSERVE_LEN - 1, :]
    )
    return model_input


def compute_ade(predicted_future, true_future):
    """Average displacement error.

    input:
        predicted_future: (N, 12, 2)
        true_future: (N, 12, 2)
    output:
        mean L2 distance across all predicted steps (float)
    """
    l2_per_step = torch.norm(predicted_future - true_future, dim=-1)
    ade = l2_per_step.mean().item()
    return ade


def compute_fde(predicted_future, true_future):
    """Final displacement error.

    input:
        predicted_future: (N, 12, 2)
        true_future: (N, 12, 2)
    output:
        mean L2 distance at the final predicted step (float)
    """
    l2_final = torch.norm(predicted_future[:, -1, :] - true_future[:, -1, :], dim=-1)
    fde = l2_final.mean().item()
    return fde


def get_all_predictions(model, test_data, device):
    """Run the model over the full test set.

    Predicted future deltas are cumsummed onto the last observed position
    so downstream code sees absolute coordinates everywhere.

    input:
        model: trained TrajectoryPredictor
        test_data: (N, 20, 2) tensor of test trajectories
        device: torch device
    output:
        (all_predictions, all_ground_truth), each (N, 20, 2) on CPU
    """
    model.eval()
    test_loader = DataLoader(TensorDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for (batch_trajectories,) in test_loader:
            batch_trajectories = batch_trajectories.to(device)
            model_input = build_model_input(batch_trajectories)
            raw_output = model(model_input)

            # rebuild absolute positions from predicted deltas
            last_observed_pos = batch_trajectories[:, OBSERVE_LEN - 1:OBSERVE_LEN, :]
            predicted_deltas = raw_output[:, OBSERVE_LEN:, :]
            predicted_future = last_observed_pos + predicted_deltas.cumsum(dim=1)

            predictions = batch_trajectories.clone()
            predictions[:, OBSERVE_LEN:, :] = predicted_future

            all_predictions.append(predictions.cpu())
            all_ground_truth.append(batch_trajectories.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    return all_predictions, all_ground_truth


def plot_examples(all_predictions, all_ground_truth, num_examples):
    """Plot predicted vs ground-truth trajectories and save a png.

    Shows the 8 observed steps, the 12 ground-truth future steps, and the
    12 predicted future steps.

    input:
        all_predictions: (N, 20, 2) absolute predicted trajectories
        all_ground_truth: (N, 20, 2) absolute ground-truth trajectories
        num_examples: how many random examples to plot
    output:
        None (writes a png to PREDICTIONS_PATH)
    """
    total = all_predictions.shape[0]
    random_seed = 41
    np.random.seed(random_seed)
    chosen_indices = np.random.choice(total, size=num_examples, replace=False)

    # roughly square grid
    n_cols = int(np.ceil(np.sqrt(num_examples)))
    n_rows = int(np.ceil(num_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for plot_idx, traj_idx in enumerate(chosen_indices):
        predicted_traj = all_predictions[traj_idx].numpy()
        true_traj = all_ground_truth[traj_idx].numpy()

        observed_xy = true_traj[:OBSERVE_LEN]
        true_future_xy = true_traj[OBSERVE_LEN:]
        pred_future_xy = predicted_traj[OBSERVE_LEN:]

        ax = axes[plot_idx]
        ax.plot(observed_xy[:, 0], observed_xy[:, 1], "ko-", label="Observed", markersize=4)
        ax.plot(true_future_xy[:, 0], true_future_xy[:, 1], "g^-", label="True future", markersize=4)
        ax.plot(pred_future_xy[:, 0], pred_future_xy[:, 1], "rs-", label="Predicted", markersize=4)

        # crosshairs at the normalization origin
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

        ax.set_title(f"Trajectory {traj_idx}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")

    # blank the leftover grid cells
    for extra_idx in range(num_examples, len(axes)):
        axes[extra_idx].axis("off")

    plt.suptitle(f"Predicted vs ground-truth trajectories - lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}", y=1.02)
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PATH, bbox_inches="tight")
    print(f"Prediction plot saved to {PREDICTIONS_PATH}")


def main():
    """Load the checkpoint, evaluate ADE/FDE, and save an example plot.

    input:
        None (reads config constants at module top)
    output:
        None (prints metrics and writes a png)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_data = torch.load(os.path.join(PROCESSED_DIR, "test.pt"))
    print(f"Test trajectories: {test_data.shape[0]}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: No checkpoint found at {CHECKPOINT_PATH}")
        print("Run train.py first to generate a checkpoint.")
        return

    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    all_predictions, all_ground_truth = get_all_predictions(model, test_data, device)

    predicted_future = all_predictions[:, OBSERVE_LEN:, :]  # (N, 12, 2)
    true_future = all_ground_truth[:, OBSERVE_LEN:, :]  # (N, 12, 2)

    ade = compute_ade(predicted_future, true_future)
    fde = compute_fde(predicted_future, true_future)

    print(f"\nResults on {SCENE} test set:")
    print(f"  ADE (avg L2 over 12 steps): {ade:.4f} m")
    print(f"  FDE (L2 at final step):     {fde:.4f} m")

    plot_examples(all_predictions, all_ground_truth, num_examples=NUM_PLOT)


if __name__ == "__main__":
    main()
