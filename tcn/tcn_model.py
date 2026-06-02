import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    """A single TCN block consisting of two dilated causal convolutions."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Causal padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        # Residual connection: match dimensions if input channels != output channels
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        # Chisel off the extra padding on the right to maintain causality
        out = out[:, :, :x.size(2)]
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TrajectoryPredictor(nn.Module):
    """TCN-based Trajectory Predictor."""
    def __init__(self, input_dim=2, output_dim=2, num_channels=[64, 64], kernel_size=3, dropout=0.0):
        super(TrajectoryPredictor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilations: 1, 2, 4, 8...
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.output_projection = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, 2)
        returns: (batch, seq_len, 2)
        """
        # 1. TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # 2. Pass through TCN blocks
        y = self.tcn(x)
        
        # 3. Project back to 2D coordinates (x, y)
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        y = y.transpose(1, 2)
        return self.output_projection(y)

if __name__ == "__main__":
    # Quick shape check
    model = TrajectoryPredictor()
    dummy_input = torch.randn(4, 20, 2)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")   # (4, 20, 2)
    print(f"Output shape: {output.shape}")        # (4, 20, 2)
