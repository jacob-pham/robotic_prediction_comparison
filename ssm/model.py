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
        "and save it as s4d.py in the project root."
    )
    sys.exit(1)

INPUT_DIM = 2
HIDDEN_DIM = 64
STATE_DIM = 64
OUTPUT_DIM = 2
NUM_LAYERS = 2
FF_EXPANSION = 2  # feedforward layer expands hidden dim by this factor


class TrajectoryPredictor(nn.Module):
    """A simple SSM-based trajectory predictor.

    Architecture (in order):
      1. Input projection: Linear(2 -> HIDDEN_DIM)
      2. NUM_LAYERS Transformer-style blocks, each containing:
           a. Pre-norm S4D sub-block:
                x = x + S4D(LayerNorm(x))
           b. Pre-norm feedforward sub-block (with GELU nonlinearity):
                x = x + Linear -> GELU -> Linear(LayerNorm(x))
         The feedforward sub-block is critical — it provides the only
         nonlinearity in the model. Without it, the whole network is
         approximately linear and cannot fit nonlinear motion patterns.
      3. Output projection: Linear(HIDDEN_DIM -> 2)

    Input:  (batch, 20, 2) — step-to-step displacements (velocities).
            Index 0 is zero (no prior frame), indices 1-7 are the
            observed deltas, indices 8-19 are zeroed (the model rolls
            out from its internal state).
    Output: (batch, 20, 2) — at timesteps 8-19, the values are step-to-step
            displacements (delta from the previous frame), not absolute
            positions. Loss is computed only on timesteps 8-19.
    """

    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(INPUT_DIM, HIDDEN_DIM)

        # Each block needs: S4D layer, two LayerNorms, and a feedforward net
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
        """
        input_sequence: (batch, 20, 2)
        returns:        (batch, 20, 2)
        """
        hidden = self.input_projection(input_sequence)

        for s4d_layer, norm_s4d, norm_ff, feedforward in zip(
            self.s4d_layers,
            self.norms_before_s4d,
            self.norms_before_ff,
            self.feedforwards,
        ):
            # Sub-block 1: S4D with pre-norm and residual
            normed = norm_s4d(hidden)
            s4d_output = s4d_layer(normed)[0]  # S4D returns (output, state)
            hidden = hidden + s4d_output

            # Sub-block 2: feedforward with pre-norm and residual
            normed = norm_ff(hidden)
            ff_output = feedforward(normed)
            hidden = hidden + ff_output

        predicted_sequence = self.output_projection(hidden)

        return predicted_sequence


def main():
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