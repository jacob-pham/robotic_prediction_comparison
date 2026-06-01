import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import TrajectoryPredictor

SCENE = 'zara2'

DATA_DIR = Path(r"/Users/adelaidegray1/Desktop/PIC 16B/datasets_processed") / SCENE

OBSERVE_LEN = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.003
NUM_EPOCHS = 50

def add_elements(data):
    """Add features to the tensors from the datasets_preprocessed folder
    Features to add: velocity (vx, vy, speed), acceleration, turning angles (angle, angle_change)
    
    input:
        data: a torch.tensor of shape (N, 20, 2)
    output:
        data_features: a torch.tensor of shape (N, 20, 8)
    """
    N = data.shape[0]

    vel = data[:, 1:, :] - data[:, :-1, :]                          # shape: (N, 19, 2)
    # update shape from (N, 19, 2) --> (N, 20, 2) by filling in spaces with zeros
    vel = torch.cat([torch.zeros(N,1,2), vel], dim=1)               # shape: (N, 20, 2)
    speed = torch.norm(vel, dim=2, keepdim=True)                    # shape: (N, 20, 1)

    acc = speed[:,1:,:] - speed[:,:-1, :]                           # shape: (N, 19, 1)
    acc = torch.cat([torch.zeros(N,1,1), acc], dim=1)               # shape: (N, 20, 1)

    ang = torch.atan2(vel[:,:,1:2], vel[:,:,0:1])                   # shape: (N, 20, 1)
    ang_change = ang[:,1:] - ang[:,:-1]                             # shape: (N, 19, 1)
    ang_change = torch.cat([torch.zeros(N,1,1), ang_change], dim=1) # shape: (N, 20, 1)

    # concatenate all of the tensors created above
    data_features = torch.cat([
        data,        # x, y
        vel,         # vx, vy
        speed,
        acc,
        ang,
        ang_change
    ], dim=2)
    
    return data_features

def compute_loss(predicted_future, observed_future):
    """compute element-wise mean squared error (MSE) on the 12 future predicted steps

    inputs:
        predicted_future: output of the model, with shape (N, 12, 2) 
        observed_future: observed future positions, with shape (N, 12, 2)
    outputs:
        MSE loss tensor
    """
    return nn.functional.mse_loss(predicted_future, observed_future)

def run_one_epoch(model, data_loader, optimizer=None):
    """run one pass through the training data

    inputs: 
        model: the TrajectoryPredictor model from model.py
        data_loader: a pytorch DataLoader on the dataset
            X_batch: tensor of shape (BATCH_SIZE,  8, 8)
            Y_batch: tensor of shape (BATCH_SIZE, 12, 2)
        optimizer: controls whether the function trains or evaluates
            (None if we are working with the testing data)
    output:
        avg_loss: the average loss across all batches in the epoch, a float
    """

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss=0

    # iterate through each batch in the dataset
    for X_batch, Y_batch in data_loader:
        if optimizer is not None:
            optimizer.zero_grad()
        
        pred = model(X_batch)
        loss = compute_loss(pred, Y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # add loss from this batch 
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def main():
    """Load the data, build the DataLoader, train the model
    
    Saves all the learned weights to a .pt
    """
    # load train, val data
    train_data = torch.load(DATA_DIR / "train.pt")
    val_data   = torch.load(DATA_DIR / "val.pt")

    train_data = add_elements(train_data)
    val_data = add_elements(val_data)

    X_train = train_data[:, :OBSERVE_LEN, :]
    Y_train = train_data[:, OBSERVE_LEN:, :2]

    X_val = val_data[:, :OBSERVE_LEN, :]
    Y_val = val_data[:, OBSERVE_LEN:, :2]

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = TrajectoryPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # run through epochs
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = run_one_epoch(model, train_loader, optimizer)
        val_loss = run_one_epoch(model, val_loader)
        tqdm.write(f'Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}')

    # save learned weights to disk
    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()