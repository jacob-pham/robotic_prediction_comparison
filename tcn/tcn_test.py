import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from tcn_model import TrajectoryPredictor

#parse scene arguments
parser = argparse.ArgumentParser(description = "Test TCN Trajectory Predictor on a specific scene.")
parser.add_argument("--scene", type = str, default = "eth", choices = ["eth", "hotel", "univ", "zara1", "zara2"],
                    help = "The dataset scene directory to test.")
args = parser.parse_args()

#dynamic paths
BASE_DATA_DIR    = Path(r"C:\Users\twinh\anaconda_projects\PIC 16B Project\datasets_processed") / args.scene
CHECKPOINT_PATH  = Path.cwd() / "tcn" / args.scene / "checkpoints" / "best_model.pt"
PREDICTIONS_PATH = Path.cwd() / f"predictions_{args.scene}.png"

OBSERVE_LEN = 8    # frames we observe (indices 0–7)
PREDICT_LEN = 12   # frames we predict (indices 8–19)
BATCH_SIZE  = 64
NUM_PLOT    = 5    # how many example trajectories to plot


def build_model_input(trajectory_tensor):
    """Zero out the 12 future timesteps, same as in tcn_train.py."""
    model_input = trajectory_tensor.clone()
    model_input[:, OBSERVE_LEN:, :] = 0.0
    return model_input


def compute_ade(predicted_future, true_future):
    """Average Displacement Error: mean L2 distance over all predicted steps."""
    l2_per_step = torch.norm(predicted_future - true_future, dim = -1)  # (N, 12)
    ade = l2_per_step.mean().item()
    return ade


def compute_fde(predicted_future, true_future):
    """Final Displacement Error: L2 distance at the very last predicted step."""
    l2_final = torch.norm(predicted_future[:, -1, :] - true_future[:, -1, :], dim=-1)
    fde = l2_final.mean().item()
    return fde


def get_all_predictions(model, test_data, device):
    """Run the model over the full test set and collect predictions."""
    model.eval()
    test_loader = DataLoader(TensorDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)

    all_predictions  = []
    all_ground_truth = []

    with torch.no_grad():
        for (batch_trajectories,) in test_loader:
            batch_trajectories = batch_trajectories.to(device)
            model_input        = build_model_input(batch_trajectories)
            predictions        = model(model_input)

            all_predictions.append(predictions.cpu())
            all_ground_truth.append(batch_trajectories.cpu())

    all_predictions  = torch.cat(all_predictions,  dim=0)  # (N, 20, 2)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)  # (N, 20, 2)
    return all_predictions, all_ground_truth


def plot_examples(all_predictions, all_ground_truth, num_examples):
    """Plot a few predicted trajectories vs ground truth and save to a file."""
    total          = all_predictions.shape[0]
    chosen_indices = np.random.choice(total, size=num_examples, replace=False)

    fig, axes = plt.subplots(1, num_examples, figsize=(4 * num_examples, 4))
    if num_examples == 1:
        axes = [axes]

    for plot_idx, traj_idx in enumerate(chosen_indices):
        predicted_traj = all_predictions[traj_idx].cpu().detach().numpy()   # (20, 2)
        true_traj      = all_ground_truth[traj_idx].cpu().detach().numpy()  # (20, 2)

        observed_xy    = true_traj[:OBSERVE_LEN]             # (8, 2)
        true_future_xy = true_traj[OBSERVE_LEN:]             # (12, 2)
        pred_future_xy = predicted_traj[OBSERVE_LEN:]        # (12, 2)

        ax = axes[plot_idx]
        ax.plot(observed_xy[:, 0],    observed_xy[:, 1],    "ko-", label = "Observed",    markersize=4)
        ax.plot(true_future_xy[:, 0], true_future_xy[:, 1], "g^-", label = "True future", markersize=4)
        ax.plot(pred_future_xy[:, 0], pred_future_xy[:, 1], "rs-", label = "Predicted",   markersize=4)

        ax.axhline(0, color = "gray", linewidth = 0.5, linestyle = "--")
        ax.axvline(0, color = "gray", linewidth = 0.5, linestyle = "--")

        ax.set_title(f"Trajectory {traj_idx}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize = 7)
        ax.set_aspect("equal")

    plt.suptitle(f"Predicted vs ground-truth trajectories ({args.scene.upper()})", y=1.02)
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PATH, bbox_inches="tight")
    print(f"Prediction plot saved to {PREDICTIONS_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing scene: {args.scene.upper()}")

    #load test data
    test_path = BASE_DATA_DIR / "test.pt"
    if not test_path.exists():
        print(f"ERROR: No test dataset found at {test_path}")
        return

    test_data = torch.load(test_path)
    print(f"Test trajectories: {test_data.shape[0]}")

    #load model checkpoint
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: No checkpoint found at {CHECKPOINT_PATH}")
        print(f"Run train.py with '--scene {args.scene}' first to generate a checkpoint.")
        return

    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    #get predictions for the whole test set
    all_predictions, all_ground_truth = get_all_predictions(model, test_data, device)

    #compute ADE and FDE
    predicted_future = all_predictions[:, OBSERVE_LEN:, :]    # (N, 12, 2)
    true_future      = all_ground_truth[:, OBSERVE_LEN:, :]   # (N, 12, 2)

    ade = compute_ade(predicted_future, true_future)
    fde = compute_fde(predicted_future, true_future)

    print(f"\nResults on {args.scene.upper()} test set:")
    print(f"  ADE (avg L2 over 12 steps): {ade:.4f} m")
    print(f"  FDE (L2 at final step):     {fde:.4f} m")

    #plot example predictions
    plot_examples(all_predictions, all_ground_truth, num_examples = NUM_PLOT)


if __name__ == "__main__":
    main()
