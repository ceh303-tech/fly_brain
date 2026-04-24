# fly_brain/torus.py
"""
RLC oscillator network on a 32×32 torus.

Adapted from TTCE engine.py:
  - Reduced from 250×250 (62,500 nodes) to 32×32 (1,024 nodes)
  - Removed PyTorch / CUDA — numpy only
  - Sparse local connectivity (8 neighbours) instead of dense tesseract mix
  - Toroidal boundary conditions preserved exactly
  - T-Diode nonlinearity kept exactly
  - Stochastic Goto Gate kept exactly

Biological analogy:
  Fly optic lobe medulla neurons — each column responds to a local patch
  of visual space. Lateral connections create wave-like population codes
  that propagate spatial information across the array.

Memory: V + I (float32, 1024 nodes) = 8 KB. Well within Pi Zero 2W budget.
"""

import math
import numpy as np
from . import config as cfg
from .config import (
    TORUS_W, TORUS_H, N_NODES, N_SUB,
    DT, TORUS_FEEDBACK, THERMODYNAMIC_PROFILES,
    IMU_REGION, SONIC_REGION,
)


class FlyBrainTorus:
    """
    Fly medulla column array - 32x32 = 1024 RLC oscillator nodes.

    Biological analog: Drosophila optic lobe medulla.

      Medulla columns: ~800 columns in each fly optic lobe, each receiving
        input from one ommatidium (compound eye facet) and projecting to
        the lobula plate.  Here each 32x32 node = one medulla column.

      Phase identity (phase_id): sets the anisotropic wave propagation axis,
        analogous to the preferred direction of direction-selective T4/T5 cells
        in each medulla arm.  The three arms of the medulla (dorsal, equatorial,
        ventral) process different parts of the visual field at ~120-degree offsets,
        matching our three phase angles exactly.

      Lateral connections: T4/T5 cells contact 3-8 adjacent columns via
        transmedullary neurons.  Here: 8-connected toroidal neighbourhood.

      RLC dynamics: approximates the graded potential / calcium dynamics
        in medulla intrinsic (Mi) neurons.  The T-diode nonlinearity mirrors
        the threshold nonlinearity of Mi1/Tm3 cells in the motion detection circuit.

      Spike (Goto gate): approximates the spiking threshold of lobula plate
        tangential cells (LPTCs) which fire discretely when a motion edge
        sweeps across their preferred direction.

    Three instances run simultaneously at 0/120/240 degree phase offsets
    (ocellus, halteres, antenna) forming the complete fly sensory array.
    """

    _TWO_PI_THIRDS = 2.0 * math.pi / 3.0

    def __init__(self, w: int = None, h: int = None,
                 profile: str = 'NORMAL', phase_id: int = 0) -> None:
        """Initialise torus state and build local neighbourhood lookup."""
        self.N = (w or TORUS_W) * (h or TORUS_H)
        self.W = w or TORUS_W
        self.H = h or TORUS_H

        # RLC parameters from thermodynamic profile
        p = THERMODYNAMIC_PROFILES[profile]
        self.R           = float(p['R'])
        self.L           = float(p['L'])
        self.C           = float(p['C'])
        self.V_CRIT      = float(p['V_CRIT'])

        self.ALPHA       = float(p['ALPHA'])
        self.BETA        = float(p['BETA'])
        self.V_TUNNEL    = float(p['V_TUNNEL'])
        self.THERMAL_NOISE = float(cfg.THERMAL_NOISE)
        self.SHOT_NOISE_AMP = float(cfg.SHOT_NOISE_AMP)
        self.GABA_LATERAL = float(cfg.GABA_LATERAL)
        self.TARGET_ACTIVITY = float(cfg.TARGET_ACTIVITY)
        self.TORUS_FEEDBACK = float(TORUS_FEEDBACK)

        # Per-node impedance arrays — built lazily after self.W/self.H are known.
        # Finalised in _build_impedance_masks() called at end of __init__.
        # Shape (N,) float32; default values = scalar R/L above.
        self._R_arr: np.ndarray = None  # set by _build_impedance_masks
        self._L_arr: np.ndarray = None  # set by _build_impedance_masks

        # Phase offsets: 0, 120, 240 degrees — same as TTCE
        self.phase_offsets = np.array(
            [0.0, self._TWO_PI_THIRDS, 2.0 * self._TWO_PI_THIRDS],
            dtype=np.float32,
        )

        # ----------------------------------------------------------------
        # Anisotropic routing weights — each phase propagates preferentially
        # along a different spatial axis (0° / 120° / 240°).
        # This gives each modality a distinct wave-propagation fingerprint:
        #   phase 0 (GPS)   → East-West axis
        #   phase 1 (IMU)   → NW-SE axis (~120°)
        #   phase 2 (SONIC) → SW-NE axis (~240°)
        # Neighbour order: N, NE, E, SE, S, SW, W, NW
        # ----------------------------------------------------------------
        self.phase_id = int(phase_id) % 3
        _dir_rad = [
            math.pi / 2,        # N
            math.pi / 4,        # NE
            0.0,                # E
            -math.pi / 4,       # SE
            -math.pi / 2,       # S
            -3.0 * math.pi / 4, # SW
            math.pi,            # W
            3.0 * math.pi / 4,  # NW
        ]
        _pref  = self.phase_id * self._TWO_PI_THIRDS
        _raw   = np.array(
            [0.5 + 0.5 * math.cos(d - _pref) for d in _dir_rad],
            dtype=np.float32,
        )
        self._dir_weights = _raw / _raw.sum()  # normalised weighted-mean

        # State tensors — flat (N_NODES,) to keep memory minimal
        # V and I are the primary RLC state; one effective oscillator per node.
        # Phase structure is encoded via phase-modulated V_in (see step()).
        self.V      = np.zeros(self.N, dtype=np.float32)   # 4 KB
        self.I      = np.zeros(self.N, dtype=np.float32)   # 4 KB
        self.spikes = np.zeros(self.N, dtype=np.float32)   # 4 KB
        self.V_in   = np.zeros(self.N, dtype=np.float32)   # 4 KB
        # Total state: 16 KB — well within 50 KB budget

        # Precompute local 8-neighbour indices for wave routing (static, 32 KB)
        self._neighbours = self._build_neighbours()

        # Build per-node impedance arrays — must come after W/H/N finalised
        self._build_impedance_masks()

        self.t: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_neighbours(self) -> np.ndarray:
        """
        Build (N_NODES, 8) array of neighbour indices.

        Each node has 8 neighbours: N, NE, E, SE, S, SW, W, NW.
        Toroidal wrapping: edge nodes connect to the opposite edge.
        This matches the toroidal boundary in TTCE router.py but extends
        to 8-connectivity (diagonal neighbours) for richer wave dynamics.

        Biological analogy: local lateral inhibition/excitation in
        the fly medulla — each column directly contacts its 6-8 neighbours.
        """
        W, H = self.W, self.H
        neighbours = np.zeros((self.N, 8), dtype=np.int32)

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

    def _build_impedance_masks(self) -> None:
        """
        Build per-node R and L arrays with region-specific impedance values.

        Impedance mismatch at sensor region boundaries creates partial wave
        reflections → standing waves → energy trapping → memory persistence.

          IMU region   (top-right,    x∈[16,32), y∈[0,16)):
            R=R_base, L=1.5×L_base — profile-relative contrast.

          SONIC region (bottom-left,  x∈[0,16),  y∈[16,32)):
            R=R_base, L=1.3×L_base — profile-relative contrast.

        GPS and FLOW regions retain default scalar R/L from thermodynamic profile.
        """
        self._R_arr = np.full(self.N, self.R, dtype=np.float32)
        self._L_arr = np.full(self.N, self.L, dtype=np.float32)

        imu_l = self.L * 1.5
        sonic_l = self.L * 1.3

        for region, r_val, l_val in (
          (IMU_REGION,   self.R, imu_l),
          (SONIC_REGION, self.R, sonic_l),
        ):
            x0, y0, x1, y1 = region
            rows = np.arange(y0, min(y1, self.H))
            cols = np.arange(x0, min(x1, self.W))
            nodes = (rows[:, None] * self.W + cols[None, :]).ravel()
            self._R_arr[nodes] = r_val
            self._L_arr[nodes] = l_val

    # ------------------------------------------------------------------
    # Core dynamics — adapted from TTCE engine.py step()
    # ------------------------------------------------------------------

    def step(self, dt: float = DT) -> np.ndarray:
        """
        Advance RLC dynamics one timestep (Euler integration).

        V_in is a persistent drive — sensors inject each step via inject().
        Routing feedback is added internally (not accumulated in V_in).

        Returns: (N_NODES,) voltage state.
        """
        t_phase = float(self.t) * dt

        # Phase envelope: mean of 3 phases (0, 120, 240 deg)
        phase_env = float(np.mean(np.sin(t_phase + self.phase_offsets)))
        V_in_eff  = self.V_in * (0.5 + 0.5 * phase_env)  # (N,)

        # Local routing: anisotropic weighted mean of 8 neighbours.
        # Each phase propagates best along its preferred spatial axis.
        neighbour_vsums = self.V[self._neighbours]              # (N, 8)
        routed          = neighbour_vsums @ self._dir_weights   # (N,)
        V_drive         = V_in_eff + self.TORUS_FEEDBACK * routed

        # ----- Euler RLC step -----
        dV = (dt / self.C) * self.I
        dI = (dt / self._L_arr) * (V_drive - self.V - self._R_arr * self.I)

        V_new = self.V + dV
        I_new = self.I + dI

        # ----- Shot noise: stochastic resonance term on current -----
        i_dc = float(np.mean(np.abs(I_new)))
        shot_sigma = self.SHOT_NOISE_AMP * math.sqrt(max(i_dc, 0.0))
        if shot_sigma > 0.0:
          I_new = I_new + np.random.randn(self.N).astype(np.float32) * shot_sigma

        # ----- Negative resistance around V_CRIT -----
        v_low = self.V_CRIT * 0.5
        v_high = self.V_CRIT * 1.5
        nr_mask = ((V_new > v_low) & (V_new < v_high)).astype(np.float32)
        I_new = I_new + self.ALPHA * (V_new - self.V_CRIT) * dt * nr_mask

        # ----- Stochastic Goto Gate (kept exactly from TTCE) -----
        noise  = np.random.randn(self.N).astype(np.float32) * self.THERMAL_NOISE
        exp_arg = -self.BETA * (V_new + noise - self.V_TUNNEL)
        P_fire = 1.0 / (1.0 + np.exp(np.clip(exp_arg, -60.0, 60.0)))
        self.spikes = (np.random.rand(self.N) < P_fire).astype(np.float32)

        # ----- Lateral inhibition: firing nodes suppress neighbouring currents -----
        firing_idx = np.flatnonzero(V_new > self.V_CRIT)
        if firing_idx.size > 0 and self.GABA_LATERAL > 0.0:
          firing_strength = self.GABA_LATERAL * V_new[firing_idx] * dt
          neigh_idx = self._neighbours[firing_idx].reshape(-1)
          neigh_suppr = np.repeat(firing_strength, self._neighbours.shape[1])
          np.add.at(I_new, neigh_idx, -neigh_suppr)

        # ----- Divisive normalization: keep global activity in range -----
        mean_activity = float(np.mean(np.abs(V_new)))
        if self.TARGET_ACTIVITY > 0.0 and mean_activity > self.TARGET_ACTIVITY:
          I_new *= self.TARGET_ACTIVITY / mean_activity

        # ----- Clamp for numerical stability -----
        np.clip(V_new, -10.0, 10.0, out=V_new)
        np.clip(I_new, -10.0, 10.0, out=I_new)

        self.V = V_new
        self.I = I_new
        self.t += 1

        # V_in persists — sensor injection accumulates until explicitly zeroed.
        # SensorInjector adds to V_in each control step. Between control steps
        # V_in is NOT cleared here; the controller.step() is responsible for
        # zeroing V_in before each new sensor injection cycle.

        return self.get_state()

    # ------------------------------------------------------------------
    # Injection interface
    # ------------------------------------------------------------------

    def inject(self, x: int, y: int, amplitude: float) -> None:
        """
        Inject voltage at torus coordinate (x, y).

        Biological analogy: a sensory afferent fibre making a synaptic
        contact at a specific location in the medulla column map.

        Adds amplitude to both V_in (persistent drive) and V (immediate
        depolarisation), so the injection has an effect within the same step.

        x: column index [0, TORUS_W)
        y: row index    [0, TORUS_H)
        amplitude: injected voltage
        """
        x = int(x) % self.W
        y = int(y) % self.H
        node = y * self.W + x
        self.V_in[node] += float(amplitude)
        self.V[node]    += float(amplitude) * 0.1   # immediate depolarisation

    def inject_region(self, region: tuple, pattern: np.ndarray) -> None:
        """
        Inject a wave pattern into a rectangular torus region.

        region: (x0, y0, x1, y1) — column/row bounds, exclusive upper.
        pattern: 2D float array with shape matching the region dimensions.

        Biological analogy: a retinotopic projection — a whole patch of
        photoreceptors driving a corresponding patch of medulla columns.
        """
        x0, y0, x1, y1 = region
        region_h = y1 - y0
        region_w = x1 - x0

        pat = np.asarray(pattern, dtype=np.float32)
        if pat.shape != (region_h, region_w):
            pat = pat.reshape(region_h, region_w)

        for ry in range(region_h):
            for rx in range(region_w):
                gx = (x0 + rx) % self.W
                gy = (y0 + ry) % self.H
                node = gy * self.W + gx
                self.V_in[node] += pat[ry, rx]
                self.V[node]    += pat[ry, rx] * 0.1  # immediate depolarisation

    # ------------------------------------------------------------------
    # State readout
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """
        Return flattened voltage state vector. Shape: (N_NODES,).

        Returns the per-node voltage (float32 copy).
        This is the 'population wave' used by MotorReadout and HebbianPlasticity.
        """
        return self.V.astype(np.float32)

    def spike_rate(self) -> float:
        """
        Return fraction of nodes currently above V_CRIT (spiking).

        Biological analogy: population firing rate — a key measure of
        how 'awake' or activated the network is. Should be < 1% at rest
        with NORMAL profile, rises with sensor injection.
        """
        return float(self.spikes.mean())

    def reset(self) -> None:
        """Zero all state tensors. Useful for test isolation."""
        self.V[:]      = 0.0
        self.I[:]      = 0.0
        self.V_in[:]   = 0.0
        self.spikes[:] = 0.0
        self.t = 0

    def memory_kb(self) -> float:
        """Return memory used by wave state arrays in KB (excludes static tables)."""
        return (self.V.nbytes + self.I.nbytes +
                self.V_in.nbytes + self.spikes.nbytes) / 1024.0
