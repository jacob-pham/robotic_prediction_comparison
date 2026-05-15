from model import build_model
import numpy as np
import torch

train_tensors = []
val_tensors = []

scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

for scene in scenes:
    train_tensors.append(torch.load(f'datasets/{scene}/train.pt'))
    val_tensors.append(torch.load(f'datasets/{scene}/val.pt'))

train_data = torch.cat(train_tensors, dim=0)
val_data = torch.cat(val_tensors, dim=0)

def add_velocity(train_data, val_data):
    """Add velocity data to each trajectory
    Args: 
    train_data: (N, 20, 2) torch.float - the last dimension is 2 for (x,y)
    val_data:   (N, 20, 2) torch.float - the last dimension is 2 for (x,y)
    Returns:
    train_data: (N, 20, 4) torch.float - the last dimension is for (x, y, vx, vy)
    val_data:   (N, 20, 4) torch.float - the last dimension is for (x, y, vx, vy)
    """
    
    train_vel = train_data[:, 1:, :] - train_data[:, :-1, :]
    val_vel = val_data[:, 1:, :] - val_data[:, :-1, :]

    zero_pad_train = torch.zeros(train_data.shape[0], 1, 2)
    zero_pad_val = torch.zeros(val_data.shape[0], 1, 2)

    train_v = torch.cat([zero_pad_train, train_vel], dim=1)
    val_v = torch.cat([zero_pad_val, val_vel], dim=1)

    train_data = torch.cat([train_data, train_v], dim=2)
    val_data = torch.cat([val_data, val_v], dim=2)

    return train_data, val_data

train_data, val_data = add_velocity(train_data, val_data)

X_train = train_data[:, :8, :].numpy()
Y_train = train_data[:, 8:, :2].numpy()

X_val = val_data[:, :8, :]
Y_val = val_data[:, 8:, :2]

model = build_model()

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,       # number of times the model sees all of the data
    batch_size=64    # number of samples processed before updating weights
)

model.save('trajectory_model.keras')
