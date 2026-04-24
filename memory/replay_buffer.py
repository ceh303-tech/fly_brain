# fly_brain/memory/replay_buffer.py
"""
Experience replay buffer for fly_brain catastrophic forgetting prevention.

Biological analogy: hippocampal replay during sleep.
The hippocampus stores compressed snapshots of recent experiences.
During offline consolidation these are replayed back to cortex
to reinforce earlier learning while incorporating new experience.

Implementation uses sparse storage — only nodes that received
significant injection above REPLAY_STORE_THRESHOLD are stored.
This exploits the fact that the torus wave state is not random:
most nodes are quiet at any moment. Only active injection regions
and wave fronts have significant voltage.

Memory cost: approximately 200 KB for 500 snapshots
(assuming ~5% node activity — 50 active nodes per snapshot)
Compare to full storage: 500 × 1024 × float32 = 2 MB
"""

import numpy as np
from collections import deque
from typing import Optional
from ..config import (N_NODES, REPLAY_BUFFER_SIZE,
                      REPLAY_STORE_THRESHOLD)


class SparseSnapshot:
    """
    Compressed representation of one torus injection state.

    Instead of storing all 1024 node voltages, stores only
    the indices and values of nodes above threshold.

    This is the sparse matrix approach — store the index of
    active points rather than the full dense array.

    For a torus with 5% activity (typical during sensor injection):
    Full storage:   1024 × float32 = 4 KB
    Sparse storage: 51 × (int16 + float16) = 204 bytes
    Compression ratio: ~20×
    """

    def __init__(self, V_in: np.ndarray, threshold: float):
        active_mask = np.abs(V_in) > threshold
        self.indices = np.where(active_mask)[0].astype(np.int16)
        self.values = V_in[active_mask].astype(np.float16)
        self.n_nodes = len(V_in)

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct full injection vector from sparse representation.
        Inactive nodes are zero — they had no significant injection.
        """
        V_in = np.zeros(self.n_nodes, dtype=np.float32)
        if len(self.indices) > 0:
            V_in[self.indices.astype(np.int32)] = self.values.astype(np.float32)
        return V_in

    @property
    def memory_bytes(self) -> int:
        """Actual memory used by this snapshot."""
        return (self.indices.nbytes + self.values.nbytes)

    @property
    def n_active(self) -> int:
        """Number of active nodes stored."""
        return len(self.indices)


class ReplayBuffer:
    """
    Circular buffer of past sensor injection states.

    Stores sparse snapshots of V_in (the torus injection vector)
    rather than full torus states. On replay, the stored injection
    pattern is re-injected into the torus and a Hebbian update
    is performed with reward=0.0.

    The key insight: V_in generated the wave state.
    Re-injecting V_in regenerates a similar wave state.
    You store the cause, not the effect.
    This is the Minecraft method — store the seed, not the world.

    Usage:
        buffer = ReplayBuffer()
        buffer.store(controller.torus.V_in)           # each step
        if buffer.ready:
            pattern = buffer.sample()                  # replay step
    """

    def __init__(self, capacity: int = None):
        self.capacity = capacity or REPLAY_BUFFER_SIZE
        self._buffer: deque = deque(maxlen=self.capacity)
        self._total_stored = 0

    def store(self, V_in: np.ndarray):
        """
        Store current injection state as sparse snapshot.
        Called every step during live flight.
        Only stores if there is significant injection activity.
        """
        snapshot = SparseSnapshot(V_in, REPLAY_STORE_THRESHOLD)
        # Only store if there is meaningful activity to replay
        if snapshot.n_active > 0:
            self._buffer.append(snapshot)
            self._total_stored += 1

    def sample(self) -> Optional[np.ndarray]:
        """
        Return a random past injection pattern as a full vector.
        Returns None if buffer is not ready.
        """
        if not self.ready:
            return None
        idx = np.random.randint(len(self._buffer))
        return self._buffer[idx].reconstruct()

    def sample_batch(self, n: int) -> list:
        """
        Return n random past patterns.
        Useful for averaging replay updates.
        """
        if not self.ready:
            return []
        indices = np.random.choice(len(self._buffer),
                                   size=min(n, len(self._buffer)),
                                   replace=False)
        return [self._buffer[i].reconstruct() for i in indices]

    @property
    def ready(self) -> bool:
        """True when buffer has enough entries to replay."""
        from ..config import REPLAY_MIN_BUFFER
        return len(self._buffer) >= REPLAY_MIN_BUFFER

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def memory_bytes(self) -> int:
        """Total memory used by stored snapshots."""
        return sum(s.memory_bytes for s in self._buffer)

    def memory_report(self) -> dict:
        return {
            'n_snapshots': len(self._buffer),
            'total_stored': self._total_stored,
            'memory_kb': self.memory_bytes / 1024,
            'mean_active_nodes': np.mean([s.n_active for s in self._buffer])
            if self._buffer else 0,
        }
