import sys
import torch
import torch.nn as nn

try:
    from s4d import S4D
except ImportError:
    print(
        "ERROR: s4d.py not found.\n"
        "Download it from:\n"
        "  https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py\n"
    )
    sys.exit(1)

INPUT_DIM = 2
HIDDEN_DIM = 64
STATE_DIM = 64
OUTPUT_DIM = 2
NUM_LAYERS = 2
FF_EXPANSION = 2  # feedforward layer expands hidden dim by this factor


class TrajectoryPredictor(nn.Module):
    """Simple SSM-based trajectory predictor.

    Architecture: Linear(2 -> HIDDEN_DIM), then NUM_LAYERS pre-norm blocks
    of (S4D + residual) and (FF GELU + residual), then Linear(HIDDEN_DIM -> 2).
    The feedforward sub-block is what gives the network its nonlinearity.
    """

    def __init__(self):
        """Build the layers.

        input:
            None
        output:
            None (initializes module parameters)
        """
        super().__init__()

        self.input_projection = nn.Linear(INPUT_DIM, HIDDEN_DIM)

        # each block: S4D, two LayerNorms, and a feedforward net
        self.s4d_layers = nn.ModuleList([
            S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0)
            for _ in range(NUM_LAYERS)
        ])
        self.norms_before_s4d = nn.ModuleList([
            nn.LayerNorm(HIDDEN_DIM) for _ in range(NUM_LAYERS)
        ])
        self.norms_before_ff = nn.ModuleList([
            nn.LayerNorm(HIDDEN_DIM) for _ in range(NUM_LAYERS)
        ])
        self.feedforwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM * FF_EXPANSION),
                nn.GELU(),
                nn.Linear(HIDDEN_DIM * FF_EXPANSION, HIDDEN_DIM),
            )
            for _ in range(NUM_LAYERS)
        ])

        self.output_projection = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, input_sequence):
        """Forward pass.

        input:
            input_sequence: (batch, 20, 2) step deltas; future slots zeroed
        output:
            (batch, 20, 2) where future slots are predicted step deltas
        """
        hidden = self.input_projection(input_sequence)

        for s4d_layer, norm_s4d, norm_ff, feedforward in zip(
            self.s4d_layers,
            self.norms_before_s4d,
            self.norms_before_ff,
            self.feedforwards,
        ):
            # S4D sub-block (pre-norm, residual)
            normed = norm_s4d(hidden)
            s4d_output = s4d_layer(normed)[0]  # S4D returns (output, state)
            hidden = hidden + s4d_output

            # feedforward sub-block (pre-norm, residual)
            normed = norm_ff(hidden)
            ff_output = feedforward(normed)
            hidden = hidden + ff_output

        predicted_sequence = self.output_projection(hidden)

        return predicted_sequence


def main():
    """Test the model on a dummy batch.

    input:
        None
    output:
        None (prints shapes and param count)
    """
    print("Running forward pass on dummy batch...")

    model = TrajectoryPredictor()
    batch_size = 4
    seq_len = 20

    dummy_input = torch.randn(batch_size, seq_len, INPUT_DIM)
    output = model(dummy_input)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("Shape check passed!")


if __name__ == "__main__":
    main()