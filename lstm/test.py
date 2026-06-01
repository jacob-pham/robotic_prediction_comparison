import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from model import TrajectoryPredictor
from train import add_elements

SCENE = 'zara2'

DATA_DIR = Path(r"/Users/adelaidegray1/Desktop/PIC 16B/datasets_processed") / SCENE

OBSERVE_LEN = 8
BATCH_SIZE = 64

def compute_ade(predicted_future, observed_future):
    """compute average displacement error (ADE)
    calculate the mean L2 distance across all predicted steps (12 steps)
    
    input:
        predicted_future: trajectory tensor of size (N, 12, 2)
        observed_future:  trajectory tensor of size (N, 12, 2)
    output:
        ade: a float
    """

    l2_per_step = torch.norm(predicted_future-observed_future, dim=-1)
    ade = l2_per_step.mean().item()
    return ade

def compute_fde(predicted_future, observed_future):
    """compute final displacement error (FDE)
    calculate the L2 distance at the final predicted step (1 steps)
    
    input:
        predicted_future: trajectory tensor of size (N, 12, 2)
        observed_future:  trajectory tensor of size (N, 12, 2)
    output:
        fde: a float
    """
    l2_final = torch.norm(predicted_future[:, -1, :] - observed_future[:, -1, :], dim=-1)
    fde = l2_final.mean().item()
    return fde

def get_all_predictions(model, data_loader, device):
    """Run the model over the full test set

    inputs:
        model: the TrajectoryPredictor model from model.py
        data_loader: a pytorch DataLoader on the dataset
            X_batch: tensor of shape (BATCH_SIZE,  8, 8)
            Y_batch: tensor of shape (BATCH_SIZE, 12, 2)
        device: either "cuda" or "cpu," determined in the main function

    outputs:
        all_preds: tensor of predicted future trajectories with shape (N, 12, 2)
        all_obs:    tensor of observed future trajectories with shape (N, 12, 2)
    """

    model.eval()
    all_preds = []
    all_obs = []

    # don't track gradients to save memory (so use .no_grad())
    with torch.no_grad():
        # iterate through each batch in data_loader
        for X_batch, Y_batch in data_loader:
            # move each batch of input data to the same device as the model is in
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            # collect predictions from each batch
            all_preds.append(preds.cpu())
            all_obs.append(Y_batch)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_obs = torch.cat(all_obs, dim=0)
    return all_preds, all_obs

def main():
    """Load the data, build the test DataLoader, load the trained model, 
       run predictions, evaluate ADE and FDE

    prints the test ADE and test FDE
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load test deta
    test_data = torch.load(DATA_DIR / "test.pt")
    test_data = add_elements(test_data)

    X_test = test_data[:, :OBSERVE_LEN, :]
    Y_test = test_data[:, OBSERVE_LEN:, :2]

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = TrajectoryPredictor().to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))

    # get predictions
    all_preds, all_obs = get_all_predictions(model, test_loader, device)

    # compute ade, fde
    ade = compute_ade(all_preds, all_obs)
    fde = compute_fde(all_preds, all_obs)

    print(f"Test ADE: {ade:.4f}")
    print(f"Test FDE: {fde:.4f}")

if __name__ == "__main__":
    main()
