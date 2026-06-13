import argparse
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tcn_model import TrajectoryPredictor

#parse scene arguments
parser = argparse.ArgumentParser(description="Train TCN Trajectory Predictor on a specific scene.")
parser.add_argument("--scene", type=str, default = "univ", choices = ["eth", "hotel", "univ", "zara1", "zara2"],
                    help = "The dataset scene directory to train on.")
args = parser.parse_args()

#dynamic paths
BASE_DATA_DIR = Path.cwd().parent / "datasets_processed" / args.scene
print(f"Using processed data from: {BASE_DATA_DIR}")
CHECKPOINT_DIR = Path.cwd() / args.scene / "checkpoints"
LOSS_CURVE_PATH = Path.cwd() / args.scene / "loss_curve.png"

#hyperparameters
OBSERVE_LEN  = 8    # frames we observe (indices 0–7)
PREDICT_LEN  = 12   # frames we predict (indices 8–19)
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS   = 50


def positions_to_velocities(traj):
    # traj: (B, T, 2)
    vel = torch.zeros_like(traj)
    vel[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
    return vel

def random_rotate(traj):
    angle = torch.rand(1) * 2 * torch.pi
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[c, -s], [s, c]])
    return traj @ R.T

def build_model_input(trajectory_tensor):
    """Zero out the future 12 timesteps so the model only sees past observations."""
    model_input = trajectory_tensor.clone()
    model_input[:, OBSERVE_LEN:, :] = 0.0   # blank out timesteps 8–19
    return model_input

def run_one_epoch(model, data_loader, optimizer, is_training, device):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss   = 0.0
    total_batches = 0

    for batch_trajectories in data_loader:
        batch_trajectories = batch_trajectories[0].to(device)
        batch_trajectories = positions_to_velocities(batch_trajectories)
        if is_training:
            batch_trajectories = random_rotate(batch_trajectories)
        model_input = build_model_input(batch_trajectories)

        if is_training:
            optimizer.zero_grad()
            predictions = model(model_input)
            loss        = compute_custom_loss(predictions, batch_trajectories)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(model_input)
                loss        = compute_custom_loss(predictions, batch_trajectories)

        total_loss    += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on scene: {args.scene.upper()}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_path = BASE_DATA_DIR / "train.pt"
    val_path = BASE_DATA_DIR / "val.pt"
    
    train_data = torch.load(train_path)
    val_data   = torch.load(val_path)
    print(f"Train trajectories: {train_data.shape[0]}  |  Val trajectories: {val_data.shape[0]}")

    train_loader = DataLoader(TensorDataset(train_data), batch_size = BATCH_SIZE, shuffle = True)
    val_loader   = DataLoader(TensorDataset(val_data),   batch_size = BATCH_SIZE, shuffle = False)

    #model, optimizer
    model     = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    #training loop
    train_losses = []
    val_losses   = []
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        train_loss = run_one_epoch(model, train_loader, optimizer, is_training = True,  device = device)
        val_loss   = run_one_epoch(model, val_loader,   optimizer, is_training = False, device = device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  |  LR: {current_lr:.6f}  |  train loss: {train_loss:.6f}  |  val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val loss {best_val_loss:.6f} — checkpoint saved.")

        scheduler.step(val_loss)

    print(f"\nTraining complete for {args.scene.upper()}. Best val loss: {best_val_loss:.6f}")

    #loss curve
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label = "Train loss")
    plt.plot(epochs, val_losses,   label = "Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"Training and validation loss ({args.scene.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
