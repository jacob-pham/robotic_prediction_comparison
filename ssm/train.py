from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import TrajectoryPredictor

# ── Hyperparameters ──────────────────────────────────────────────────────────
PROCESSED_DIR   = Path.cwd().parent / "datasets_processed" / "eth"
CHECKPOINT_DIR  = Path.cwd() / "checkpoints"

OBSERVE_LEN  = 8    # frames we observe (indices 0–7)
PREDICT_LEN  = 12   # frames we predict (indices 8–19)
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
NUM_EPOCHS   = 100

SAVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_best_model.pt"
LOSS_CURVE_PATH = CHECKPOINT_DIR / f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_loss_curve.png"


def build_model_input(trajectory_tensor):
    """Zero out the future 12 timesteps so the model only sees past observations.

    trajectory_tensor: (N, 20, 2) — full ground-truth trajectories
    returns:           (N, 20, 2) — observed steps kept, future steps set to 0
    """
    model_input = trajectory_tensor.clone()
    model_input[:, OBSERVE_LEN:, :] = 0.0   # blank out timesteps 8–19
    return model_input


def compute_loss(predictions, ground_truth):
    """MSE loss computed only on the 12 predicted future timesteps (indices 8–19).

    predictions:  (batch, 20, 2) — full model output
    ground_truth: (batch, 20, 2) — full ground-truth trajectory
    """
    predicted_future = predictions[:, OBSERVE_LEN:, :]     # (batch, 12, 2)
    true_future      = ground_truth[:, OBSERVE_LEN:, :]    # (batch, 12, 2)
    return nn.functional.mse_loss(predicted_future, true_future)


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

    total_loss   = 0.0
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

        total_loss    += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_data = torch.load(PROCESSED_DIR / "train.pt").to(device)
    val_data   = torch.load(PROCESSED_DIR / "val.pt").to(device)
    print(f"Train trajectories: {train_data.shape[0]}  |  Val trajectories: {val_data.shape[0]}")

    # TensorDataset wraps a tensor so DataLoader can iterate over it in batches
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_data),   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, optimizer ─────────────────────────────────────────────────────
    model     = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []
    best_val_loss = float("inf")

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training")
    for epoch in pbar:
        train_loss = run_one_epoch(model, train_loader, optimizer, is_training=True,  device=device)
        val_loss   = run_one_epoch(model, val_loader,   optimizer, is_training=False, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"Epoch {epoch}: train = {train_loss:.4f}, val = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            # tqdm.write(f"  new best val loss: {best_val_loss:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    # ── Loss curve ───────────────────────────────────────────────────────────
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses,   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"Training and validation loss - lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_epochs{NUM_EPOCHS}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
