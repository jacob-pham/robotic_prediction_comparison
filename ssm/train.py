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

SCENE = "zara2"  # scene to make test set, available scenes: "eth", "hotel", "univ", "zara1", "zara2"
VERSION = "v2"  # model version, refer to git commit history 

PROCESSED_DIR = Path.cwd().parent / "datasets_processed" / SCENE
print(f"Using processed data from: {PROCESSED_DIR}")
CHECKPOINT_DIR = Path.cwd() / SCENE / VERSION

OBSERVE_LEN = 8 # frames we observe (indices 0–7)
PREDICT_LEN = 12 # frames we predict (indices 8–19)

SAVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_best_model.pt"
LOSS_CURVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_loss_curve.png"


def build_model_input(trajectory_tensor):
    """Turn positions into the step-delta model input.

    The model learns to extrapolate recent velocity, not raw positions.

    input:
        trajectory_tensor: (N, 20, 2) ground-truth positions
    output:
        (N, 20, 2) with observed deltas at indices 1..OBSERVE_LEN-1 and
        zeros elsewhere
    """
    model_input = torch.zeros_like(trajectory_tensor)
    # observed deltas: pos[t] - pos[t-1]
    model_input[:, 1:OBSERVE_LEN, :] = (
        trajectory_tensor[:, 1:OBSERVE_LEN, :] - trajectory_tensor[:, :OBSERVE_LEN - 1, :]
    )
    return model_input


def compute_loss(predictions, ground_truth):
    """MSE on step deltas over the 12 future timesteps.

    Future model output is interpreted as deltas, so the target at index t
    is gt[t] - gt[t-1].

    input:
        predictions: (batch, 20, 2) model output (future slots are deltas)
        ground_truth: (batch, 20, 2) absolute positions
    output:
        scalar MSE loss tensor
    """
    predicted_deltas = predictions[:, OBSERVE_LEN:, :]
    target_deltas = ground_truth[:, OBSERVE_LEN:, :] - ground_truth[:, OBSERVE_LEN - 1:-1, :]
    return nn.functional.mse_loss(predicted_deltas, target_deltas)

def check_for_stopping_criterion(epoch, val_loss, val_losses):
    """Stop if val loss hasn't improved by 1% in the last 10 epochs.

    input:
        epoch: current epoch (1-indexed)
        val_loss: this epoch's val loss
        val_losses: list of all val losses so far (including this one)
    output:
        True if we should stop, else False
    """
    criterion = 10  # epochs of no improvement allowed

    if epoch < criterion + 1:
        return False

    prior_val_losses = val_losses[-criterion - 1:-1]
    if not prior_val_losses:
        return False
    best_prior_val_loss = min(prior_val_losses)
    relative_improvement = (best_prior_val_loss - val_loss) / best_prior_val_loss
    return relative_improvement < 0.01

def run_one_epoch(model, data_loader, optimizer, is_training, device):
    """One full pass over the data.

    input:
        model: TrajectoryPredictor
        data_loader: DataLoader yielding (trajectory_tensor,) tuples
        optimizer: torch optimizer (only used when is_training=True)
        is_training: True for train, False for val
        device: torch device (unused; tensors come from data_loader)
    output:
        average loss across batches (float)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch_trajectories in data_loader:
        # DataLoader wraps each batch in a tuple; unpack the tensor
        batch_trajectories = batch_trajectories[0]

        model_input = build_model_input(batch_trajectories)

        if is_training:
            optimizer.zero_grad()
            predictions = model(model_input)
            loss = compute_loss(predictions, batch_trajectories)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(model_input)
                loss = compute_loss(predictions, batch_trajectories)

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss


def main():
    """Train the model and save the best checkpoint plus a loss curve.

    input:
        None (reads config constants at module top)
    output:
        None (writes a .pt checkpoint and a .png loss curve)
    """
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
