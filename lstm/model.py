import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    def __init__(self):
        """LSTM trajectory predictor

        Layers:
            LSTM: two stacked LSTMs with input_size=8 for the 8 features. hidden_size=8
            Linear: inputs 8 numbers, outputs 24 for 12 predictedsteps * 2 coordinates
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=2, batch_first=True)
        self.fc = nn.Linear(8, 24)

    def forward(self, x):
        """Forward pass

        input:
            x: input sequences (N, 8, 8)
        output:
            x_pred: (N, 12, 2)
        """
        output, _ = self.lstm(x)
        x = output[:,-1,:] 
        x = self.fc(x)
        x = x.reshape(-1, 12, 2)
        return x
