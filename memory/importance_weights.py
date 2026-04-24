# fly_brain/memory/importance_weights.py
"""
Elastic Weight Consolidation adapted for Hebbian networks.

In standard EWC (Kirkpatrick et al. 2017), importance scores are computed
from the Fisher information matrix — expensive and requires backpropagation.

For Hebbian networks, importance is approximated by cumulative eligibility:
connections that have fired together frequently are important.
Changing them degrades past performance.

Biological analogy: synaptic tagging and capture.
Synapses that have been repeatedly potentiated become tagged as important
and resist depotentiation. The molecular tag marks the synapse for
consolidation — it becomes harder to overwrite.

Usage:
    importance = ImportanceTracker(weights_shape)
    importance.update(delta_W)          # each Hebbian update
    penalty = importance.ewc_penalty(weights)  # each update step
"""

import numpy as np
from ..config import (N_NODES, CONNECTIONS_PER_NODE,
                      EWC_IMPORTANCE_LR, EWC_IMPORTANCE_DECAY,
                      EWC_CONSOLIDATION_STRENGTH)


class ImportanceTracker:
    """
    Tracks which Hebbian connections are important for past behaviour.
    Generates EWC penalty that resists overwriting important connections.

    The importance score for connection (i,j) accumulates when that
    connection fires (high |delta_W|) and decays slowly when unused.
    High importance → strong resistance to change → protects past learning.
    """

    def __init__(self, n_nodes: int = None, n_connections: int = None):
        self.n_nodes = n_nodes or N_NODES
        self.n_connections = n_connections or CONNECTIONS_PER_NODE
        # Importance scores — initialised to zero (no prior importance)
        self.importance = np.zeros(
            (self.n_nodes, self.n_connections), dtype=np.float32)

    def update(self, delta_W: np.ndarray):
        """
        Update importance scores based on recent weight change.
        Connections that changed a lot are marked as important.
        Importance decays slowly over time.

        delta_W: shape (n_nodes, n_connections) — recent Hebbian update
        """
        # Accumulate importance from recent activity
        self.importance += EWC_IMPORTANCE_LR * np.abs(delta_W)
        # Slow decay — important connections retain their tag
        self.importance *= (1.0 - EWC_IMPORTANCE_DECAY)
        # Clip to reasonable range
        self.importance = np.clip(self.importance, 0.0, 1.0)

    def ewc_penalty(self, weights: np.ndarray,
                    anchor_weights: np.ndarray) -> np.ndarray:
        """
        Compute EWC penalty for current weight update.

        The penalty resists moving important weights away from
        their anchor values (values at environment boundary).

        penalty[i,j] = importance[i,j] × (weights[i,j] - anchor[i,j])

        High importance + large deviation = large penalty.
        Low importance = small penalty, weight can change freely.

        weights:        current weights
        anchor_weights: weights at the point when importance was computed
                        (typically saved when switching environments)
        """
        return (EWC_CONSOLIDATION_STRENGTH
                * self.importance
                * (weights - anchor_weights))

    def consolidate(self, weights: np.ndarray):
        """
        Snapshot current weights as the anchor point.
        Call this when the drone enters a new environment.
        Returns the anchor weight array for storage.
        """
        return weights.copy()

    @property
    def mean_importance(self) -> float:
        return float(np.mean(self.importance))

    @property
    def memory_bytes(self) -> int:
        return self.importance.nbytes
