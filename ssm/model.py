import sys
import torch
import torch.nn as nn

# ── S4D import ────────────────────────────────────────────────────────────
# s4d.py must be in the project root. If it's missing, print a clear message.
try:
    from s4d import S4D
except ImportError:
    print(
        "ERROR: s4d.py not found.\n"
        "Download it from:\n"
        "  https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py\n"
        "and save it as s4d.py in the project root."
    )
    sys.exit(1)

# ── Architecture constants ────────────────────────────────────────────────
INPUT_DIM   = 2    # x and y coordinates
HIDDEN_DIM  = 64   # width of the SSM layers
STATE_DIM   = 64   # internal state size inside each S4D layer
OUTPUT_DIM  = 2    # predict x and y
NUM_LAYERS  = 2    # how many stacked S4D blocks


class TrajectoryPredictor(nn.Module):
    """A simple SSM-based trajectory predictor.

    Architecture (in order):
      1. A linear layer projects the 2D input to HIDDEN_DIM features.
      2. NUM_LAYERS S4D blocks, each with a pre-norm residual connection:
            x = x + S4D(LayerNorm(x))
         This is the standard "pre-norm" transformer-style residual pattern.
      3. A linear layer projects from HIDDEN_DIM back to 2D output positions.

    Input:
      A tensor of shape (batch, 20, 2).
      Timesteps 0-7 contain observed positions; timesteps 8-19 are zero.

    Output:
      A tensor of shape (batch, 20, 2).
      Loss is only computed on timesteps 8-19 (the 12 predicted steps).
    """

    def __init__(self):
        super().__init__()

        # Project from 2D coordinates into the model's hidden dimension
        self.input_projection = nn.Linear(INPUT_DIM, HIDDEN_DIM)

        # Build NUM_LAYERS separate S4D layers and LayerNorms.
        # We store them in ModuleLists so PyTorch tracks their parameters.
        self.s4d_layers  = nn.ModuleList([
            # transposed=False tells S4D to expect (batch, seq_len, d_model)
            # instead of the default (batch, d_model, seq_len) channels-first format.
            S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0)
            for _ in range(NUM_LAYERS)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(HIDDEN_DIM)
            for _ in range(NUM_LAYERS)
        ])

        # Project from hidden dimension back to 2D predicted positions
        self.output_projection = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, input_sequence):
        """
        input_sequence: (batch, 20, 2) — observed steps filled, future steps zeroed
        returns:        (batch, 20, 2) — full sequence of predicted positions
        """
        # (batch, 20, 2) -> (batch, 20, HIDDEN_DIM)
        hidden = self.input_projection(input_sequence)

        # Apply each S4D block with pre-norm residual connection
        for s4d_layer, layer_norm in zip(self.s4d_layers, self.layer_norms):
            # Pre-norm: normalize first, then pass through S4D
            normed = layer_norm(hidden)

            # S4D returns a tuple (output, state); we only need the output
            s4d_output = s4d_layer(normed)[0]

            # Residual connection: add the block's input back to its output
            hidden = hidden + s4d_output

        # (batch, 20, HIDDEN_DIM) -> (batch, 20, 2)
        predicted_sequence = self.output_projection(hidden)

        return predicted_sequence


if __name__ == "__main__":
    # Quick shape check: run a single dummy batch through the model
    print("Running forward pass on dummy batch...")

    model      = TrajectoryPredictor()
    batch_size = 4
    seq_len    = 20

    # Random input: (batch, 20, 2)
    dummy_input = torch.randn(batch_size, seq_len, INPUT_DIM)
    output      = model(dummy_input)

    print(f"  Input shape:  {dummy_input.shape}")   # expect (4, 20, 2)
    print(f"  Output shape: {output.shape}")         # expect (4, 20, 2)
    print("Shape check passed!")
