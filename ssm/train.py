from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import TrajectoryPredictor

BATCH_SIZE = 512
LEARNING_RATE = 0.003
NUM_EPOCHS = 150

SCENE = "univ"  # scene to make test set, available scenes: "eth", "hotel", "univ", "zara1", "zara2"
VERSION = "v2"  # model version, refer to git commit history 

PROCESSED_DIR = Path.cwd().parent / "datasets_processed" / SCENE
print(f"Using processed data from: {PROCESSED_DIR}")
CHECKPOINT_DIR = Path.cwd() / SCENE / VERSION

OBSERVE_LEN = 8 # frames we observe (indices 0–7)
PREDICT_LEN = 12 # frames we predict (indices 8–19)

SAVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_best_model.pt"
LOSS_CURVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_loss_curve.png"


def build_model_input(trajectory_tensor):
    """Build the model input from a full trajectory tensor.

    Feeds step-to-step displacements (velocities) at the observed
    timesteps instead of raw positions. The model's job is then to
    extrapolate the recent velocity rather than infer it from positions.

    trajectory_tensor: (N, 20, 2) — full ground-truth trajectories (positions)
    returns:           (N, 20, 2) — observed deltas at indices 1..OBSERVE_LEN-1,
                                    zeros at index 0 and indices OBSERVE_LEN..19
    """
    model_input = torch.zeros_like(trajectory_tensor)
    # Observed deltas: position[t] - position[t-1] for t in 1..OBSERVE_LEN-1
    model_input[:, 1:OBSERVE_LEN, :] = (
        trajectory_tensor[:, 1:OBSERVE_LEN, :] - trajectory_tensor[:, :OBSERVE_LEN - 1, :]
    )
    return model_input


def compute_loss(predictions, ground_truth):
    """MSE loss on step-to-step displacements for the 12 future timesteps.

    The model output at indices 8-19 is interpreted as the displacement
    from the previous frame, not an absolute position. So the target at
    index t is ground_truth[t] - ground_truth[t-1] for t in 8..19.

    predictions:  (batch, 20, 2) — full model output (future slots are deltas)
    ground_truth: (batch, 20, 2) — full ground-truth trajectory (positions)
    """
    predicted_deltas = predictions[:, OBSERVE_LEN:, :]  # (batch, 12, 2)
    target_deltas = ground_truth[:, OBSERVE_LEN:, :] - ground_truth[:, OBSERVE_LEN - 1:-1, :]  # (batch, 12, 2)
    return nn.functional.mse_loss(predicted_deltas, target_deltas)

def check_for_stopping_criterion(epoch, val_loss, val_losses):
    """Example stopping criterion: stop if val loss doesn't improve for {criterion} epochs."""
    criterion = 10  # how many epochs of no improvement to tolerate before stopping

    if epoch < criterion + 1:
        return False  # Don't stop in the first criterion epochs

    # if val_loss has not improved by at least 1% compared to the best
    # val loss in the previous {criterion} epochs, stop
    prior_val_losses = val_losses[-criterion - 1:-1]
    if not prior_val_losses:
        return False
    best_prior_val_loss = min(prior_val_losses)
    relative_improvement = (best_prior_val_loss - val_loss) / best_prior_val_loss
    return relative_improvement < 0.01

def run_one_epoch(model, data_loader, optimizer, is_training, device):
    """Run one full pass over the data (train or val).

    If is_training=True, computes gradients and updates weights.
    If is_training=False, just computes loss without changing anything.
    Returns the average loss across all batches.
    """
    if is_training:
        model.train()   # turns on dropout etc. (none here, but good practice)
    else:
        model.eval()    # turns off dropout etc.

    total_loss = 0.0
    total_batches = 0

    for batch_trajectories in data_loader:
        # DataLoader wraps each batch in a list; unpack the single tensor
        batch_trajectories = batch_trajectories[0]

        model_input = build_model_input(batch_trajectories)  # zero out future

        if is_training:
            optimizer.zero_grad()                       # clear old gradients
            predictions = model(model_input)            # forward pass
            loss        = compute_loss(predictions, batch_trajectories)
            loss.backward()                             # compute new gradients
            optimizer.step()                            # update weights
        else:
            with torch.no_grad():                       # no gradient tracking needed
                predictions = model(model_input)
                loss        = compute_loss(predictions, batch_trajectories)

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = torch.load(PROCESSED_DIR / "train.pt").to(device)
    val_data = torch.load(PROCESSED_DIR / "val.pt").to(device)
    print(f"Train trajectories: {train_data.shape[0]}  |  Val trajectories: {val_data.shape[0]}")

    # TensorDataset wraps a tensor so DataLoader can iterate over it in batches
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training")
    for epoch in pbar:
        train_loss = run_one_epoch(model, train_loader, optimizer, is_training=True, device=device)
        val_loss = run_one_epoch(model, val_loader, optimizer, is_training=False, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"Epoch {epoch}: train = {train_loss:.4f}, val = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            # tqdm.write(f"  new best val loss: {best_val_loss:.6f}")
        # if check_for_stopping_criterion(epoch, val_loss, val_losses):
        #     tqdm.write(f"Stopping early at epoch {epoch} due to no significant improvement in val loss.")
        #     break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"Training and validation loss - lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs{NUM_EPOCHS}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
