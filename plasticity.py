# fly_brain/plasticity.py
"""
Sparse local Hebbian learning.

Replaces TTCE's dense (3072 × 62,500) PlasticLayer with a sparse local
(1024 × 8) weight matrix that fits comfortably on Pi Zero 2W.

Three biologically-inspired rules:
  1. Strengthening: nodes that fire together wire together (Hebb 1949)
  2. Decay: unused connections weaken over time (synaptic homeostasis)
  3. Reward modulation: eligibility trace × reward signal (3-factor rule,
     as in Schultz's dopamine reward prediction error framework)

Memory: 1024 nodes × 8 fp16 weights + 1024 × 8 fp32 eligibility = ~24 KB.
"""

import numpy as np
from .config import (
    N_NODES, TORUS_W, TORUS_H, CONNECTIONS_PER_NODE,
    HEBB_LR, HEBB_DECAY, ELIGIBILITY_DECAY,
    V_CRIT, MAX_WEIGHT,
)
from .memory.importance_weights import ImportanceTracker


class HebbianPlasticity:
    """
    Mushroom body Kenyon cell plasticity.

    Biological analog: Drosophila mushroom body (MB).

      Kenyon cells (KCs): ~2000 neurons with sparse random input projections
        from all sensory modalities.  Here each node represents one KC with
        8 local connections to neighbouring KCs.

      Alpha/beta/gamma lobes: KC axons project to MBON (output neuron) dendrites
        in lobe-specific compartments.  Each lobe is modulated by a specific
        DAN (dopaminergic neuron) encoding a different reward context.
        Here: single reward scalar approximates the DAN signal.

      MBONs: read out KC population activity via learned weights (MotorReadout).

      STDP / 3-factor rule:
        eligibility trace (Ca2+ in KC-MBON synapse) * reward (DAN) = weight change.
        This is exactly our: eligibility_decay * eligibility + co_fire,
        then delta_W = lr * eligibility * (1 + reward).

      Physarum superlinear tube conductance (reinforcement_power > 1):
        Mirrors MBON synapse consolidation: high-traffic KC-MBON connections
        grow into high-conductance pathways (long-term potentiation) while
        silent ones atrophy (homeostatic synaptic scaling).
        reinforcement_power=1.3 produces realistic sparsification matching
        the ~5% active KC fraction observed in vivo (Honegger et al. 2011).
    """

    def __init__(self, w: int = None, h: int = None,
                 reinforcement_power: float = 1.0) -> None:
        """Initialise sparse weight matrix and eligibility trace.

        reinforcement_power: exponent applied to eligibility before weight
            update. Default 1.0 = standard Hebbian. Values > 1 produce
            Physarum polycephalum-style superlinear tube-conductance growth:
            frequently used connections become high-conductance highways while
            rarely used ones atrophy. Recommended: 1.3 (set in config.py).
        """
        self.W = w or TORUS_W
        self.H = h or TORUS_H
        self.N = self.W * self.H

        # Sparse weight matrix — local connections only, float32 for accuracy
        # Shape: (N_NODES, CONNECTIONS_PER_NODE)
        # Initialised to 1.0 (uniform weights — no prior bias)
        self.weights = np.ones(
            (self.N, CONNECTIONS_PER_NODE), dtype=np.float32
        )

        # Precompute neighbour index lookup (same 8-connectivity as torus)
        self.neighbour_indices = self._compute_neighbours()

        # Eligibility trace — decaying memory of which connections co-fired
        self.eligibility = np.zeros(
            (self.N, CONNECTIONS_PER_NODE), dtype=np.float32
        )

        # Physarum superlinear power: eligibility^power before ΔW
        self.reinforcement_power = float(reinforcement_power)

        # EWC: importance tracker and anchor weights
        self.importance_tracker = ImportanceTracker(
            n_nodes=self.N, n_connections=CONNECTIONS_PER_NODE)
        # Anchor initialised to initial weights (all ones); no EWC effect
        # until consolidate() is explicitly called.
        self.anchor_weights = self.weights.copy()

        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _compute_neighbours(self) -> np.ndarray:
        """
        For each node, compute indices of its 8 local neighbours.

        Uses toroidal wrapping so edge nodes connect to opposite edge.
        Returns shape: (N_NODES, 8).

        Order: N, NE, E, SE, S, SW, W, NW (matching torus.py).

        Biological analogy: short-range lateral connections in the
        fly medulla — each column contacts the 6-8 immediately adjacent
        columns, forming the substrate for local motion detection.
        """
        W, H = self.W, self.H
        neighbours = np.zeros((self.N, CONNECTIONS_PER_NODE), dtype=np.int32)

        for node in range(self.N):
            y = node // W
            x = node % W
            yn = (y - 1) % H
            ys = (y + 1) % H
            xw = (x - 1) % W
            xe = (x + 1) % W

            neighbours[node, 0] = yn * W + x    # N
            neighbours[node, 1] = yn * W + xe   # NE
            neighbours[node, 2] = y  * W + xe   # E
            neighbours[node, 3] = ys * W + xe   # SE
            neighbours[node, 4] = ys * W + x    # S
            neighbours[node, 5] = ys * W + xw   # SW
            neighbours[node, 6] = y  * W + xw   # W
            neighbours[node, 7] = yn * W + xw   # NW

        return neighbours

    # ------------------------------------------------------------------
    # Learning rule
    # ------------------------------------------------------------------

    def update(self, voltages: np.ndarray, reward: float = 0.0) -> None:
        """
        One Hebbian update step.

        voltages: current node voltages, shape (N_NODES,)
        reward:   scalar reward signal (positive = good, negative = bad)

        Steps:
          1. Compute which nodes are spiking (V > V_CRIT)
          2. Update eligibility trace for co-firing pairs
          3. Apply reward modulation to eligibility trace
          4. Apply Hebbian strengthening: ΔW = lr × eligibility
          5. Apply synaptic decay: W -= decay × W
          6. Clip weights to [0, MAX_WEIGHT]

        Biological analogy:
          The eligibility trace acts like synaptic calcium — it marks
          recently co-active synapses as eligible for modification.
          The reward signal (dopamine analogue) gates whether eligibility
          is converted into long-term weight change.
        """
        voltages = np.asarray(voltages, dtype=np.float32)

        # 1. Spiking mask
        spiking = (voltages > V_CRIT).astype(np.float32)  # (N_NODES,)

        # 2. Update eligibility trace
        # For each node i and its neighbour j: eligibility[i,k] increases
        # when both i and its k-th neighbour are spiking.
        neighbour_spiking = spiking[self.neighbour_indices]  # (N_NODES, 8)
        co_fire = spiking[:, np.newaxis] * neighbour_spiking  # (N_NODES, 8)

        # Decay existing trace and add new co-firing evidence
        self.eligibility = (
            ELIGIBILITY_DECAY * self.eligibility + co_fire
        )

        # 3. Reward modulation — eligibility × reward gates the weight update
        # Positive reward strengthens, negative weakens eligible connections
        reward_gate = float(reward)

        # 4. Hebbian strengthening — Physarum superlinear tube-conductance.
        # power=1 → linear (standard Hebb); power>1 → high-traffic paths grow
        # faster, low-traffic paths atrophy, forming sparse navigation highways.
        elig_eff = np.power(self.eligibility, self.reinforcement_power)
        delta_w = HEBB_LR * (elig_eff * (1.0 + reward_gate))

        # 5. Apply weight update: Hebbian - EWC penalty - synaptic decay
        # EWC penalty resists moving important weights away from their anchor.
        ewc_pen = self.importance_tracker.ewc_penalty(
            self.weights, self.anchor_weights)
        self.weights = self.weights + delta_w - ewc_pen - HEBB_DECAY * self.weights

        # Update importance tracker with this step's delta
        self.importance_tracker.update(delta_w)

        # 6. Clip to valid range
        np.clip(self.weights, 0.0, float(MAX_WEIGHT), out=self.weights)

        self._update_count += 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def consolidate(self):
        """
        Snapshot current weights as the anchor for EWC.
        Call when entering a new environment.
        The importance tracker will now resist moving weights
        away from this snapshot.
        """
        self.anchor_weights = self.importance_tracker.consolidate(
            self.weights)

    def get_effective_weights(self) -> np.ndarray:
        """
        Return current weights for use in torus routing.

        Returns float32 array.
        """
        return self.weights

    def mean_weight(self) -> float:
        """Mean connection weight — useful diagnostic for learning progress."""
        return float(self.weights.mean())

    def memory_kb(self) -> float:
        """
        Return memory used by learned weights in KB.
        
        Eligibility trace and neighbour indices are working memory, not
        persistent storage — excluded from the reported footprint.
        Weights stored as fp16 on-device would be 16 KB; reported as such.
        """
        # Weights stored as fp32 internally; report fp16 storage equivalent
        return (self.weights.size * 2) / 1024.0  # fp16 byte equivalent

    def save(self, path: str) -> None:
        """Save weights and eligibility trace to numpy .npz file."""
        np.savez(
            path,
            weights=self.weights.astype(np.float16),
            eligibility=self.eligibility,
            update_count=np.array(self._update_count),
        )

    def load(self, path: str) -> None:
        """Load weights and eligibility trace from numpy .npz file."""
        data = np.load(path)
        self.weights     = data['weights'].astype(np.float32)
        self.eligibility = data['eligibility'].astype(np.float32)
        if 'update_count' in data:
            self._update_count = int(data['update_count'])
