# fly_brain/controller.py
"""
FlyBrainController - biologically-grounded three-phase wave navigation controller.

Biological architecture (Drosophila melanogaster):
===============================================================

PHASE 0  torus_ocellus   (+1 trit pole)
  Biological analog: Dorsal ocelli (three simple eyes pointing skyward).
  In flies, ocelli detect wide-field luminance changes and horizon tilt.
  Here they encode CLEARANCE: how far is the nearest obstacle in each direction.
  High amplitude = obstacle far = open sky = safe to fly.  Trit pole = +1.

PHASE 1  torus_halteres  ( 0 trit pole)
  Biological analog: Halteres - modified hindwings acting as gyroscopes.
  Halteres ONLY detect angular velocity (Coriolis forces during rotation).
  They do NOT sense linear acceleration - that is Johnston's organ (antenna).
  Here they receive pure gyro signal: body rotation dynamics.  Trit pole = 0.
  Linear (accel) signal is routed to torus_antenna as Johnston's organ analog.

PHASE 2  torus_antenna   (-1 trit pole)
  Biological analog: Compound eye + Johnston's organ (antenna base).
  Compound eyes detect obstacle motion and proximity (near-field optic flow).
  Johnston's organ detects near-field air vibration / linear acceleration.
  Here they receive PROXIMITY: how close is the nearest obstacle.
  High amplitude = obstacle close = danger.  Trit pole = -1.

LOBULA PLATE (ψ interference bus)
  Biological analog: Lobula plate tangential cells (LPTCs) in the optic lobe.
  LPTCs are wide-field motion detectors that integrate signals across ALL
  visual columns.  They receive input from both the compound eye (motion)
  and the halteres (rotation stabilisation via neck motor neurons).
  Here: ψ = A·e^{i0} + B·e^{i2π/3} + C·e^{i4π/3} computed across all nodes.
  |ψ| is high when ONE phase fires uniquely (novel event) and collapses to
  zero when all three balance (background consensus).  Leaky-integrated to
  give working-memory duration matching LPTC integration time (~50 ms).

MUSHROOM BODY (Hebbian plasticity + motor readout)
  Biological analog: Mushroom body Kenyon cells → MBONs, gated by DANs.
  Kenyon cells (KCs): sparse random projections from all sensory modalities.
  MBONs (output neurons): linear readout of KC population.
  DANs (dopaminergic neurons): reward prediction error = our 'reward' scalar.
  Physarum superlinear rule mimics KC-to-MBON synapse consolidation:
  frequently co-active KC-MBON pairs grow into high-conductance pathways
  (like MBON axon potentiation), while silent ones atrophy (homeostasis).

BALANCED TERNARY READOUT (descending neurons)
  Biological analog: Descending neurons (DNs) - ~350 cells that read out
  from mushroom body / central complex and drive thoracic motor circuits.
  trit = V_in_ocellus - V_in_antenna
    > 0  clearance > proximity  -> safe    -> reward +1 region
    < 0  proximity > clearance  -> danger  -> reward -1 region
    = 0  no sensor data or balanced
  This signal is DIRECTLY correlated with reward without any training;
  the delta-rule readout converges in tens of steps.
"""

import math
import os
import numpy as np
from .torus      import FlyBrainTorus
from .sensors    import SensorInjector
from .plasticity import HebbianPlasticity
from .readout    import MotorReadout
from .memory.replay_buffer import ReplayBuffer
from .config     import (
    DT, N_MOTORS,
    CROSS_PHASE_COUPLING, SR_NOISE_AMP, COHERENCE_DECAY, SLIME_MOLD_POWER,
    REPLAY_EVERY_N_STEPS, EWC_CONSOLIDATION_STRENGTH, READOUT_EWC_STRENGTH,
)

# Lobula plate phasor constants (120 degree separation)
_COS120 = math.cos(2.0 * math.pi / 3.0)   # -0.5
_SIN120 = math.sin(2.0 * math.pi / 3.0)   # +0.8660254
_COS240 = math.cos(4.0 * math.pi / 3.0)   # -0.5
_SIN240 = math.sin(4.0 * math.pi / 3.0)   # -0.8660254


class FlyBrainController:
    """
    Three-phase Drosophila-inspired wave navigation controller.

    Three 32x32 RLC oscillator tori run at 120 degree mutual phase offset,
    each handling one sensory modality with dedicated Hebbian connections
    (the 3-phase power rule: each wire belongs to exactly one phase).

    The lobula-plate phasor bus couples all three tori each step via
    constructive/destructive quantum-amplitude interference.

    Total memory: ~130 KB (within 200 KB budget for Pi Zero 2W).
    """

    def __init__(self, torus_w: int = None, torus_h: int = None,
                 profile: str = 'NORMAL', use_replay: bool = True) -> None:
        w = torus_w
        h = torus_h
        n = (w or 32) * (h or 32)

        # ---- Three dedicated tori - biologically named ----
        # Ocellus torus:  +1 trit pole - dorsal ocelli - clearance encoding
        self.torus_ocellus  = FlyBrainTorus(w=w, h=h, profile=profile, phase_id=0)
        # Halteres torus:  0 trit pole - gyroscopic organs - rotation dynamics
        self.torus_halteres = FlyBrainTorus(w=w, h=h, profile=profile, phase_id=1)
        # Antenna torus:  -1 trit pole - compound eye + Johnston organ - proximity
        self.torus_antenna  = FlyBrainTorus(w=w, h=h, profile=profile, phase_id=2)

        # Backward-compat aliases used by existing tests
        self.torus       = self.torus_ocellus    # ctrl.torus -> ocellus
        self.torus_safe  = self.torus_ocellus
        self.torus_imu   = self.torus_halteres
        self.torus_sonic = self.torus_antenna

        self.sensors = SensorInjector()

        # ---- Mushroom body: per-phase Kenyon cell Hebbian plasticity ----
        # Each torus has its own KC population - connections are phase-private.
        # Physarum superlinear power = KC-to-MBON synapse consolidation.
        self.plasticity_ocellus  = HebbianPlasticity(w=w, h=h,
                                                     reinforcement_power=SLIME_MOLD_POWER)
        self.plasticity_halteres = HebbianPlasticity(w=w, h=h,
                                                     reinforcement_power=SLIME_MOLD_POWER)
        self.plasticity_antenna  = HebbianPlasticity(w=w, h=h,
                                                     reinforcement_power=SLIME_MOLD_POWER)

        # Backward-compat aliases
        self.plasticity       = self.plasticity_ocellus
        self.plasticity_safe  = self.plasticity_ocellus
        self.plasticity_imu   = self.plasticity_halteres
        self.plasticity_sonic = self.plasticity_antenna

        # ---- Motor readout: descending neurons (MBONs -> DNs) ----
        self.readout = MotorReadout(n_nodes=n)

        # ---- Lobula plate phasor bus (leaky-integrated) ----
        # psi = A*e^{i0} + B*e^{i2pi/3} + C*e^{i4pi/3}
        self._lobula_cos = np.zeros(n, dtype=np.float32)
        self._lobula_sin = np.zeros(n, dtype=np.float32)
        self._lobula_mag = np.zeros(n, dtype=np.float32)  # |psi| per node

        # Backward-compat psi aliases
        self._psi_cos = self._lobula_cos
        self._psi_sin = self._lobula_sin
        self._psi_mag = self._lobula_mag

        self.step_count: int = 0

        # ---- Experience replay — catastrophic forgetting prevention ----
        self.replay_buffer = ReplayBuffer() if use_replay else None
        self._replay_step_count = 0
        # Readout anchor for EWC consolidation (set by consolidate_memory())
        self._readout_anchor: np.ndarray = None

    def step(
        self,
        pos_ned:         np.ndarray = None,
        pos_uncertainty: float      = 5.0,
        accel:           np.ndarray = None,
        gyro:            np.ndarray = None,
        ranges:          np.ndarray = None,
        bearings:        np.ndarray = None,
        reward:          float      = 0.0,
        dt:              float      = DT,
        inject_sensors:  bool       = True,
    ) -> np.ndarray:
        """
        One 100 Hz fly-brain control cycle.

        Sensor routing (biology-matched):
          ranges / bearings  -> torus_ocellus (clearance) + torus_antenna (proximity)
          gyro               -> torus_halteres (halteres: rotation only)
          accel              -> torus_antenna  (Johnston organ: linear motion)
          pos_ned            -> not injected by default (GPS database opt-in)

        Steps:
          1. Clear injection buffers
          2. Sensory injection (ocellus/halteres/antenna)
          3. Lobula plate feedback (psi_mag from previous step)
          4. Stochastic resonance cross-injection
          5. Advance all three tori (RLC + T-diode + Goto gate)
          6. Lobula plate phasor integration
          7. Mushroom body Hebbian update (per-phase Kenyon cells)
          8. Balanced ternary readout via descending neurons
        """
        # ---- 1. Clear injection buffers ----
        self.torus_ocellus.V_in[:]  = 0.0
        self.torus_halteres.V_in[:] = 0.0
        self.torus_antenna.V_in[:]  = 0.0

        # ---- 2. Sensory injection ----
        if inject_sensors:
            if ranges is not None:
                b_arr = np.asarray(
                    bearings if bearings is not None else np.zeros(len(ranges)),
                    dtype=np.float32)
                r_arr = np.asarray(ranges, dtype=np.float32)
                # Ocellus (+1 pole): clearance map - amplitude proportional to range
                # High amplitude = obstacle far = clear path = safe (trit > 0)
                self.sensors.inject_clearance(self.torus_ocellus, r_arr, b_arr)
                # Antenna (-1 pole): proximity + Johnston organ
                # High amplitude = obstacle close = danger (trit < 0)
                self.sensors.inject_ultrasonic(self.torus_antenna, r_arr, b_arr)

            if gyro is not None:
                # Halteres (0 pole): GYRO ONLY - halteres are pure rotation sensors.
                # Linear acceleration (accel) goes to Johnston organ (antenna torus).
                self.sensors.inject_halteres(
                    self.torus_halteres,
                    np.asarray(gyro, dtype=np.float32),
                )
            if accel is not None:
                # Johnston organ: linear acceleration -> antenna torus
                # (accel adds to proximity signal - body deceleration near obstacle)
                self.sensors.inject_johnston(
                    self.torus_antenna,
                    np.asarray(accel, dtype=np.float32),
                )

        if reward != 0.0:
            self.sensors.inject_reward(self.torus_ocellus,  reward)
            self.sensors.inject_reward(self.torus_halteres, reward)
            self.sensors.inject_reward(self.torus_antenna,  reward)

        # ---- 3. Lobula plate feedback ----
        # |psi| from previous step reinjects into all tori.
        # Nodes with high |psi| (one phase fired uniquely) get boosted;
        # nodes with |psi|~0 (background consensus) remain quiet.
        self.torus_ocellus.V_in  += CROSS_PHASE_COUPLING * self._lobula_mag
        self.torus_halteres.V_in += CROSS_PHASE_COUPLING * self._lobula_mag
        self.torus_antenna.V_in  += CROSS_PHASE_COUPLING * self._lobula_mag

        # ---- 4. Stochastic resonance (Douglass et al. 1993) ----
        # Ocellus-phase thermal noise injected push-pull into halteres/antenna.
        # Helps each torus detect sub-threshold signals it would miss alone.
        sr_noise = (np.random.randn(self.torus_ocellus.N).astype(np.float32)
                    * SR_NOISE_AMP)
        self.torus_halteres.V_in += sr_noise
        self.torus_antenna.V_in  -= sr_noise   # push-pull

        # ---- 5. Advance all three tori ----
        state_ocellus  = self.torus_ocellus.step(dt)
        state_halteres = self.torus_halteres.step(dt)
        state_antenna  = self.torus_antenna.step(dt)

        # ---- 6. Lobula plate phasor integration ----
        # Project spike vectors onto complex unit circle at 0/120/240 degrees:
        #   psi = A*e^{i*0} + B*e^{i*2pi/3} + C*e^{i*4pi/3}
        # |psi| collapses to 0 when all three phases are equal (background);
        # peaks when one phase fires uniquely (novel event detected).
        # Leaky integration gives ~50 ms working-memory (LPTC integration time).
        spikes_ocellus  = self.torus_ocellus.spikes
        spikes_halteres = self.torus_halteres.spikes
        spikes_antenna  = self.torus_antenna.spikes

        new_cos = (spikes_ocellus
                   + spikes_halteres * _COS120
                   + spikes_antenna  * _COS240)
        new_sin = (spikes_halteres * _SIN120
                   + spikes_antenna  * _SIN240)

        self._lobula_cos = (COHERENCE_DECAY * self._lobula_cos
                            + (1.0 - COHERENCE_DECAY) * new_cos)
        self._lobula_sin = (COHERENCE_DECAY * self._lobula_sin
                            + (1.0 - COHERENCE_DECAY) * new_sin)
        np.sqrt(self._lobula_cos ** 2 + self._lobula_sin ** 2,
                out=self._lobula_mag)
        # Keep psi aliases in sync
        self._psi_cos = self._lobula_cos
        self._psi_sin = self._lobula_sin
        self._psi_mag = self._lobula_mag

        # ---- 7. Balanced ternary readout (descending neurons) ----
        # Read V_in NOW — before plasticity modulation taints it.
        # trit = V_in_ocellus - V_in_antenna
        #   > 0  clearance > proximity  -> safe direction  -> reward +1
        #   < 0  proximity > clearance  -> obstacle ahead  -> reward -1
        #   = 0  uninjected nodes       -> no sensor data
        # lobula_mag boosts nodes where exactly one phase fired (salient event).
        trit_state    = self.torus_ocellus.V_in - self.torus_antenna.V_in
        readout_input = trit_state + 0.1 * self._lobula_mag
        motors = self.readout.forward(readout_input)

        if reward != 0.0:
            self.readout.update(reward)
            # EWC restoring force on readout: resist overwriting consolidated mapping.
            # Applied only when an anchor exists (after consolidate_memory() call).
            # READOUT_EWC_STRENGTH >> EWC_CONSOLIDATION_STRENGTH because
            # READOUT_LR >> HEBB_LR; proportional protection is needed.
            if self._readout_anchor is not None:
                readout_dev = self.readout.W - self._readout_anchor
                self.readout.W -= READOUT_EWC_STRENGTH * readout_dev

        # ---- 8. Mushroom body Hebbian update ----
        # Each phase has its own KC population (phase-private connections).
        # Physarum superlinear rule: high-traffic KC-MBON synapses consolidate.
        # reward = DAN (dopamine neuron) signal gating eligibility -> weight change.
        # Note: modulation is written to V_in AFTER the readout so it does not
        # corrupt the sensor-driven trit.  V_in is cleared at the top of the
        # next step() call anyway, so this residual has no lasting effect.
        for torus, plast, state in (
            (self.torus_ocellus,  self.plasticity_ocellus,  state_ocellus),
            (self.torus_halteres, self.plasticity_halteres, state_halteres),
            (self.torus_antenna,  self.plasticity_antenna,  state_antenna),
        ):
            plast.update(state, reward)
            weights    = plast.get_effective_weights()           # (N, 8)
            nbr_v      = state[plast.neighbour_indices]          # (N, 8)
            w_sum      = weights.sum(axis=1).clip(1e-6)
            modulation = (weights * nbr_v).sum(axis=1) / w_sum  # (N,)
            torus.V_in += 0.01 * modulation

        # ---- 9. Experience replay — reinforce past injection patterns ----
        if self.replay_buffer is not None:
            self.replay_buffer.store(self.torus.V_in.copy())

            self._replay_step_count += 1
            if (self._replay_step_count % REPLAY_EVERY_N_STEPS == 0
                    and self.replay_buffer.ready):
                past_pattern = self.replay_buffer.sample()
                if past_pattern is not None:
                    # Save current injection state
                    saved_V_in = self.torus.V_in.copy()
                    # Re-inject past pattern
                    self.torus.V_in[:] = past_pattern
                    # Run one torus step with past pattern
                    past_state = self.torus.step(dt)
                    # Hebbian update with no reward — reinforces past patterns
                    self.plasticity.update(past_state, reward=0.0)
                    # Restore current injection state
                    self.torus.V_in[:] = saved_V_in

        self.step_count += 1
        return motors

    def memory_footprint(self) -> dict:
        """
        Report memory usage in KB.
        Budget: 200 KB (Pi Zero 2W with 3-phase architecture).
        """
        kb_torus = sum(t.memory_kb() for t in [
            self.torus_ocellus, self.torus_halteres, self.torus_antenna
        ])
        kb_plasticity = sum(p.memory_kb() for p in [
            self.plasticity_ocellus, self.plasticity_halteres, self.plasticity_antenna
        ])
        kb_readout = self.readout.memory_kb()
        kb_lobula = (self._lobula_cos.nbytes + self._lobula_sin.nbytes
                     + self._lobula_mag.nbytes) / 1024.0
        kb_total = kb_torus + kb_plasticity + kb_readout + kb_lobula
        return {
            "torus_kb":      round(kb_torus, 1),
            "plasticity_kb": round(kb_plasticity, 1),
            "readout_kb":    round(kb_readout, 1),
            "total_kb":      round(kb_total, 1),
        }

    def consolidate_memory(self):
        """
        Consolidate current weights as anchor for EWC.
        Call this when the drone moves to a new environment.
        After calling, the EWC penalty will resist overwriting
        the current learned behaviour.
        """
        self.plasticity_ocellus.consolidate()
        self.plasticity_halteres.consolidate()
        self.plasticity_antenna.consolidate()
        # Also anchor readout weights so they resist square-training overwriting
        self._readout_anchor = self.readout.W.copy()

    def memory_report(self) -> dict:
        """Report memory usage including replay buffer."""
        base = self.memory_footprint()
        if self.replay_buffer is not None:
            replay = self.replay_buffer.memory_report()
            base['replay_buffer_kb'] = round(replay['memory_kb'], 1)
            base['replay_snapshots'] = replay['n_snapshots']
            base['replay_mean_active_nodes'] = round(
                float(replay['mean_active_nodes']), 1)
        else:
            base['replay_buffer_kb'] = 0.0
            base['replay_snapshots'] = 0
            base['replay_mean_active_nodes'] = 0.0
        return base

    def save_state(self, path: str) -> None:
        """Save learned weights: per-phase plasticity .npz + readout .npy."""
        os.makedirs(path, exist_ok=True)
        self.plasticity_ocellus.save(
            os.path.join(path, "plasticity_ocellus.npz"))
        self.plasticity_halteres.save(
            os.path.join(path, "plasticity_halteres.npz"))
        self.plasticity_antenna.save(
            os.path.join(path, "plasticity_antenna.npz"))
        self.readout.save(os.path.join(path, "readout.npy"))

    def load_state(self, path: str) -> None:
        """Load learned weights from a directory saved by save_state()."""
        self.plasticity_ocellus.load(
            os.path.join(path, "plasticity_ocellus.npz"))
        self.plasticity_halteres.load(
            os.path.join(path, "plasticity_halteres.npz"))
        self.plasticity_antenna.load(
            os.path.join(path, "plasticity_antenna.npz"))
        self.readout.load(os.path.join(path, "readout.npy"))
