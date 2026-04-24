# fly_brain/readout.py
"""
Linear readout from wave state to motor commands.

Maps the 1,024-dimensional torus wave state to 4 motor throttle commands
via a learned linear projection. Weights are updated online using a
reward-weighted regression rule (REINFORCE-style, no backpropagation).

Biological analogy:
  Fly descending neurons (DNs) — a small population of cells (~350 in flies)
  reads out from the optic lobe and central complex to generate motor commands.
  Here we use a 1024→4 linear map as the DN population.

Memory: 1024 × 4 × fp32 = 16 KB.
"""

import numpy as np
from .config import N_MOTORS, N_NODES, READOUT_LR


class MotorReadout:
    """
    Linear readout: wave state → motor commands.

    Weights trained online via reward-weighted regression.
    No backpropagation — purely local reward signal.

    Memory: 1024 × 4 × fp32 = 16 KB.
    """

    def __init__(self, n_nodes: int = None) -> None:
        """Initialise readout weights (small random, near-zero)."""
        self._n = n_nodes or N_NODES
        rng = np.random.default_rng(seed=42)
        self.W = rng.standard_normal((N_MOTORS, self._n)).astype(np.float32) * 0.01

        # Keep only the most recent state/output for immediate single-step learning
        self.last_state:  np.ndarray = None
        self.last_output: np.ndarray = None

        # Running mean output for baseline subtraction (REINFORCE)
        self._mean_output = np.zeros(N_MOTORS, dtype=np.float32)
        self._mean_alpha  = 0.05   # EMA coefficient for mean tracking

    def forward(self, wave_state: np.ndarray) -> np.ndarray:
        """
        Compute motor commands from wave state.

        wave_state: shape (N_NODES,)
        Returns: shape (N_MOTORS,) with values in [0, 1]

        Biological analogy: descending neuron population decodes the
        integrated wave activity into graded motor commands. The sigmoid-like
        clamp (via np.clip) prevents saturation — like plateau potentials
        in DN axons.
        """
        wave_state = np.asarray(wave_state, dtype=np.float32)
        raw = self.W @ wave_state                     # (N_MOTORS,)
        output = np.clip(raw, 0.0, 1.0).astype(np.float32)

        # Store only the most recent for immediate single-step learning
        self.last_state = wave_state.copy()
        self.last_output = output.copy()

        # Update running mean output
        self._mean_output = (
            (1.0 - self._mean_alpha) * self._mean_output
            + self._mean_alpha * output
        )

        return output

    def update(self, reward: float) -> None:
        """
        Update readout weights via immediate single-step REINFORCE.

        Maps reward ∈ [-1, 1] to a target motor level:
          reward = +1 → target = 1.0  (full thrust — good state)
          reward =  0 → target = 0.5  (neutral hover)
          reward = -1 → target = 0.0  (reduce thrust — bad state)

        Error = target − output, gradient = error × state × |reward|.
        Single-step update: uses ONLY the most recent state/output when
        reward arrives, avoiding the temporal credit-assignment problem.

        This is true REINFORCE (Williams 1992): each reward directly
        modulates the gradient of the immediately preceding action state.

        Biological analogy: reward prediction error (dopamine) directly
        modulates DN synaptic strength via weight change at the time of
        reward arrival, using only the synapses that were active moments before.
        """
        if self.last_state is None or self.last_output is None:
            return

        reward = float(reward)

        # Target: linearly map reward ∈ [-1, 1] → target ∈ [0, 1]
        target = np.full(N_MOTORS, 0.5 + 0.5 * reward, dtype=np.float32)

        # Single-step error
        error = target - self.last_output  # (N_MOTORS,)

        # Single-step weight update
        # ΔW[m, n] = lr × |reward| × error[m] × state[n]
        delta_W = READOUT_LR * abs(reward) * (error[:, np.newaxis] * self.last_state[np.newaxis, :])

        self.W += delta_W
        # No clipping on W — the motor output is clipped in forward()

    def memory_kb(self) -> float:
        """Return total memory used in KB."""
        return self.W.nbytes / 1024.0

    def save(self, path: str) -> None:
        """Save readout weights to numpy .npy file."""
        np.save(path, self.W)

    def load(self, path: str) -> None:
        """Load readout weights from numpy .npy file."""
        self.W = np.load(path).astype(np.float32)
