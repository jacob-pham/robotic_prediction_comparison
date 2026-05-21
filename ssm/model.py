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

        self.input_projection = nn.Linear(INPUT_DIM, HIDDEN_DIM) # project input to hidden dimension

        self.norm1_s4d = nn.LayerNorm(HIDDEN_DIM) # pre-norm for S4D layer; normalizes across feature dimension
        self.s4d1 = S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0) # first S4D layer; processes sequence and captures temporal dependencies
        self.norm1_ff = nn.LayerNorm(HIDDEN_DIM) # pre-norm for feedforward layer; normalizes across feature dimension
        self.ff1 = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * FF_EXPANSION),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM * FF_EXPANSION, HIDDEN_DIM),
        ) # first feedforward layer; adds nonlinearity and allows for complex feature interactions

        # second block
        self.norm2_s4d = nn.LayerNorm(HIDDEN_DIM) 
        self.s4d2 = S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0)
        self.norm2_ff = nn.LayerNorm(HIDDEN_DIM)
        self.ff2 = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * FF_EXPANSION),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM * FF_EXPANSION, HIDDEN_DIM),
        )

        # #third block
        # self.norm3_s4d = nn.LayerNorm(HIDDEN_DIM)
        # self.s4d3 = S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0)
        # self.norm3_ff = nn.LayerNorm(HIDDEN_DIM)
        # self.ff3 = nn.Sequential(
        #     nn.Linear(HIDDEN_DIM, HIDDEN_DIM * FF_EXPANSION),
        #     nn.GELU(),
        #     nn.Linear(HIDDEN_DIM * FF_EXPANSION, HIDDEN_DIM),
        # )

        # #fourth block
        # self.norm4_s4d = nn.LayerNorm(HIDDEN_DIM)
        # self.s4d4 = S4D(d_model=HIDDEN_DIM, d_state=STATE_DIM, transposed=False, dropout=0.0)
        # self.norm4_ff = nn.LayerNorm(HIDDEN_DIM)
        # self.ff4 = nn.Sequential(
        #     nn.Linear(HIDDEN_DIM, HIDDEN_DIM * FF_EXPANSION),
        #     nn.GELU(),
        #     nn.Linear(HIDDEN_DIM * FF_EXPANSION, HIDDEN_DIM),
        # )

        self.output_projection = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, input_sequence):
        """Forward pass.

        input:
            input_sequence: (batch, 20, 2) step deltas; future slots zeroed
        output:
            (batch, 20, 2) where future slots are predicted step deltas
        """
        hidden = self.input_projection(input_sequence)

        hidden = hidden + self.s4d1(self.norm1_s4d(hidden))[0]
        hidden = hidden + self.ff1(self.norm1_ff(hidden))

        hidden = hidden + self.s4d2(self.norm2_s4d(hidden))[0]
        hidden = hidden + self.ff2(self.norm2_ff(hidden))

        # hidden = hidden + self.s4d3(self.norm3_s4d(hidden))[0]
        # hidden = hidden + self.ff3(self.norm3_ff(hidden))

        # hidden = hidden + self.s4d4(self.norm4_s4d(hidden))[0]
        # hidden = hidden + self.ff4(self.norm4_ff(hidden))

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
