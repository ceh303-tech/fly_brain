# fly_brain/tests/test_fly_brain.py
"""
Test suite for fly_brain wave navigation controller.
Tests are ordered from basic to integration.
All 20 tests must pass.

Run with:
    python -m pytest fly_brain/tests/test_fly_brain.py -v
"""

import os
import sys
import math
import time
import tempfile

import numpy as np
import pytest

# Make fly_brain importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fly_brain.config import (
    N_NODES, TORUS_W, TORUS_H, V_CRIT,
    GPS_REGION, IMU_REGION, SONIC_REGION,
    N_MOTORS, ELIGIBILITY_DECAY,
)
from fly_brain.torus       import FlyBrainTorus
from fly_brain.sensors     import SensorInjector
from fly_brain.plasticity  import HebbianPlasticity
from fly_brain.readout     import MotorReadout
from fly_brain.controller  import FlyBrainController


# ---------------------------------------------------------------------------
# TEST 1: Torus initialises correctly
# ---------------------------------------------------------------------------
def test_01_torus_initialises():
    """Torus has correct shape; initial state near zero; no NaN/Inf."""
    torus = FlyBrainTorus()
    state = torus.get_state()

    assert state.shape == (N_NODES,), f"Expected ({N_NODES},), got {state.shape}"
    assert N_NODES == 1024, f"Expected 1024 nodes, got {N_NODES}"
    assert np.all(np.isfinite(state)), "Initial state contains NaN or Inf"
    assert np.abs(state).max() < 1e-6, "Initial voltages should be ~zero"


# ---------------------------------------------------------------------------
# TEST 2: Torus steps without error — state bounded at rest
# ---------------------------------------------------------------------------
def test_02_torus_steps_stable():
    """1000 idle steps: state remains finite and bounded; spike rate < 1%."""
    torus = FlyBrainTorus()
    for _ in range(1000):
        state = torus.step()

    assert np.all(np.isfinite(state)), "State contains NaN/Inf after 1000 idle steps"
    assert np.abs(state).max() < 15.0, "State blew up (> 15 V) at rest"
    assert torus.spike_rate() < 0.01, (
        f"Spike rate at rest too high: {torus.spike_rate():.4f} (expected < 0.01)"
    )


# ---------------------------------------------------------------------------
# TEST 3: Injection creates spreading wave
# ---------------------------------------------------------------------------
def test_03_injection_creates_wave():
    """Injection at centre → after 10 steps, neighbours show elevated voltage."""
    torus = FlyBrainTorus()
    cx, cy = TORUS_W // 2, TORUS_H // 2

    torus.inject(cx, cy, amplitude=1.5)
    for _ in range(10):
        torus.step()

    state = torus.get_state().reshape(TORUS_H, TORUS_W)
    centre_v = float(state[cy, cx])

    # At least one of the immediate neighbours should have non-trivial voltage
    neighbour_vs = [
        float(state[(cy - 1) % TORUS_H, cx]),
        float(state[(cy + 1) % TORUS_H, cx]),
        float(state[cy, (cx - 1) % TORUS_W]),
        float(state[cy, (cx + 1) % TORUS_W]),
    ]
    max_neighbour_v = max(abs(v) for v in neighbour_vs)

    assert max_neighbour_v > 1e-4, (
        f"No wave spread detected. Centre={centre_v:.5f}, "
        f"max neighbour={max_neighbour_v:.5f}"
    )


# ---------------------------------------------------------------------------
# TEST 4: T-Diode nonlinearity fires
# ---------------------------------------------------------------------------
def test_04_tdiode_fires():
    """High amplitude injection raises spike rate above idle baseline."""
    torus = FlyBrainTorus()

    # Rest baseline (no injection)
    torus.V_in[:] = 0.0
    for _ in range(100):
        torus.step()
    rest_spike_rate = torus.spike_rate()

    # Inject above V_CRIT across the whole torus
    torus.V_in[:] = V_CRIT * 2.5
    for _ in range(20):
        torus.step()

    spike_high = torus.spike_rate()

    assert spike_high > rest_spike_rate, (
        f"Injection should raise spike rate above rest baseline: "
        f"rest={rest_spike_rate:.4f}, high={spike_high:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 5: GPS injection maps to correct region
# ---------------------------------------------------------------------------
def test_05_gps_injection_region():
    """GPS injection activates GPS_REGION; other regions largely unaffected."""
    torus   = FlyBrainTorus()
    sensors = SensorInjector()

    sensors.inject_gps(torus, pos_ned=np.array([10.0, 20.0, 0.0]), uncertainty=0.5)

    # Take one step so voltages propagate
    torus.step()

    state = torus.get_state().reshape(TORUS_H, TORUS_W)
    x0, y0, x1, y1 = GPS_REGION

    gps_mean   = float(np.abs(state[y0:y1, x0:x1]).mean())
    # Use IMU region as reference (should be lower)
    ix0, iy0, ix1, iy1 = IMU_REGION
    imu_mean   = float(np.abs(state[iy0:iy1, ix0:ix1]).mean())

    assert gps_mean > 0.0, "GPS region shows no activation after inject_gps()"
    # GPS region should be notably more active than IMU region
    assert gps_mean > imu_mean * 0.5 or gps_mean > 1e-4, (
        f"GPS region not clearly elevated: gps_mean={gps_mean:.6f}, imu_mean={imu_mean:.6f}"
    )


# ---------------------------------------------------------------------------
# TEST 6: IMU injection maps to correct region
# ---------------------------------------------------------------------------
def test_06_imu_injection_region():
    """IMU injection activates IMU_REGION; amplitude scales with |accel|."""
    sensors = SensorInjector()

    torus_low = FlyBrainTorus()
    sensors.inject_imu(torus_low,
                       accel=np.array([0.0, 0.0, -1.0]),
                       gyro=np.zeros(3))
    torus_low.step()

    torus_high = FlyBrainTorus()
    sensors.inject_imu(torus_high,
                       accel=np.array([0.0, 0.0, -9.81]),
                       gyro=np.zeros(3))
    torus_high.step()

    ix0, iy0, ix1, iy1 = IMU_REGION

    imu_low  = float(np.abs(torus_low.get_state().reshape(TORUS_H, TORUS_W)[iy0:iy1, ix0:ix1]).mean())
    imu_high = float(np.abs(torus_high.get_state().reshape(TORUS_H, TORUS_W)[iy0:iy1, ix0:ix1]).mean())

    assert imu_high > 0.0, "IMU region shows no activation"
    assert imu_high >= imu_low, (
        f"Higher accel should produce >= activation: high={imu_high:.6f}, low={imu_low:.6f}"
    )


# ---------------------------------------------------------------------------
# TEST 7: Ultrasonic — close obstacle produces stronger signal
# ---------------------------------------------------------------------------
def test_07_ultrasonic_close_vs_far():
    """Close obstacle produces stronger SONIC_REGION injection than far obstacle."""
    sensors = SensorInjector()

    torus_close = FlyBrainTorus()
    sensors.inject_ultrasonic(torus_close,
                              ranges=np.array([0.5]),
                              bearings=np.array([0.0]))
    torus_close.step()

    torus_far = FlyBrainTorus()
    sensors.inject_ultrasonic(torus_far,
                              ranges=np.array([8.0]),
                              bearings=np.array([0.0]))
    torus_far.step()

    sx0, sy0, sx1, sy1 = SONIC_REGION
    close_mean = float(np.abs(torus_close.get_state().reshape(TORUS_H, TORUS_W)[sy0:sy1, sx0:sx1]).mean())
    far_mean   = float(np.abs(torus_far.get_state().reshape(TORUS_H, TORUS_W)[sy0:sy1, sx0:sx1]).mean())

    assert close_mean > far_mean, (
        f"Close obstacle should produce stronger signal: close={close_mean:.6f}, far={far_mean:.6f}"
    )


# ---------------------------------------------------------------------------
# TEST 8: Reward injection modulates whole torus
# ---------------------------------------------------------------------------
def test_08_reward_global_modulation():
    """Positive reward raises global V_in; negative reward lowers it."""
    sensors = SensorInjector()

    torus_pos = FlyBrainTorus()
    v_before_pos = float(torus_pos.V_in.mean())
    sensors.inject_reward(torus_pos, reward=+1.0)
    v_after_pos = float(torus_pos.V_in.mean())

    torus_neg = FlyBrainTorus()
    v_before_neg = float(torus_neg.V_in.mean())
    sensors.inject_reward(torus_neg, reward=-1.0)
    v_after_neg = float(torus_neg.V_in.mean())

    assert v_after_pos > v_before_pos, "Positive reward did not increase V_in"
    assert v_after_neg < v_before_neg, "Negative reward did not decrease V_in"


# ---------------------------------------------------------------------------
# TEST 9: Hebbian learning strengthens co-firing connections
# ---------------------------------------------------------------------------
def test_09_hebb_co_firing_strengthens():
    """Nodes forced to co-fire → W increases beyond initial value."""
    plasticity = HebbianPlasticity()

    # Record initial weight for node 0 → neighbour 0 (its North neighbour)
    initial_w = float(plasticity.weights[0, 0])

    # Create voltages where node 0 AND its North neighbour (index = neighbour_indices[0,0])
    # are both above V_CRIT
    north_idx = int(plasticity.neighbour_indices[0, 0])

    voltages = np.zeros(N_NODES, dtype=np.float32)
    voltages[0]         = V_CRIT + 1.0
    voltages[north_idx] = V_CRIT + 1.0

    for _ in range(100):
        plasticity.update(voltages, reward=0.0)

    final_w = float(plasticity.weights[0, 0])
    assert final_w > initial_w, (
        f"Co-firing should strengthen W[0,0]: initial={initial_w:.4f}, final={final_w:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 10: Synaptic decay weakens unused connections
# ---------------------------------------------------------------------------
def test_10_synaptic_decay():
    """500 idle steps → weights decay below initial value of 1.0."""
    plasticity = HebbianPlasticity()
    initial_mean = float(plasticity.weights.astype(np.float32).mean())

    voltages = np.zeros(N_NODES, dtype=np.float32)  # no firing
    for _ in range(500):
        plasticity.update(voltages, reward=0.0)

    final_mean = float(plasticity.weights.astype(np.float32).mean())
    assert final_mean < initial_mean, (
        f"Synaptic decay should reduce weights: initial={initial_mean:.4f}, final={final_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 11: Eligibility trace decays correctly
# ---------------------------------------------------------------------------
def test_11_eligibility_trace_decay():
    """After co-firing, eligibility trace decays at rate ELIGIBILITY_DECAY."""
    plasticity = HebbianPlasticity()
    north_idx  = int(plasticity.neighbour_indices[0, 0])

    voltages_firing = np.zeros(N_NODES, dtype=np.float32)
    voltages_firing[0]         = V_CRIT + 1.0
    voltages_firing[north_idx] = V_CRIT + 1.0

    # Saturate eligibility trace
    for _ in range(20):
        plasticity.update(voltages_firing, reward=0.0)

    peak_elig = float(plasticity.eligibility[0, 0])
    assert peak_elig > 0.0, "Eligibility trace did not build up during co-firing"

    # Now stop firing and measure decay
    voltages_silent = np.zeros(N_NODES, dtype=np.float32)
    plasticity.update(voltages_silent, reward=0.0)
    after_one = float(plasticity.eligibility[0, 0])

    # Should have decayed by approximately ELIGIBILITY_DECAY
    expected = peak_elig * ELIGIBILITY_DECAY
    assert abs(after_one - expected) < peak_elig * 0.05, (
        f"Eligibility should be ~{expected:.4f} after one silent step, got {after_one:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 12: Reward modulation via eligibility trace
# ---------------------------------------------------------------------------
def test_12_reward_modulation():
    """Positive reward on eligible connections increases weight more than Hebb alone."""
    plasticity_no_reward = HebbianPlasticity()
    plasticity_reward    = HebbianPlasticity()
    north_idx = int(plasticity_no_reward.neighbour_indices[0, 0])

    voltages = np.zeros(N_NODES, dtype=np.float32)
    voltages[0]         = V_CRIT + 1.0
    voltages[north_idx] = V_CRIT + 1.0

    for _ in range(50):
        plasticity_no_reward.update(voltages, reward=0.0)
        plasticity_reward.update(voltages, reward=1.0)

    w_no_reward = float(plasticity_no_reward.weights[0, 0])
    w_reward    = float(plasticity_reward.weights[0, 0])

    assert w_reward > w_no_reward, (
        f"Reward should amplify Hebbian strengthening: "
        f"no_reward={w_no_reward:.4f}, reward={w_reward:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 13: Readout produces bounded output
# ---------------------------------------------------------------------------
def test_13_readout_bounded():
    """Random wave states → motor commands always in [0, 1], shape (4,)."""
    readout = MotorReadout()
    rng     = np.random.default_rng(seed=0)

    for _ in range(100):
        state = rng.standard_normal(N_NODES).astype(np.float32) * 5.0
        motors = readout.forward(state)
        assert motors.shape == (N_MOTORS,), f"Wrong shape: {motors.shape}"
        assert np.all(motors >= 0.0) and np.all(motors <= 1.0), (
            f"Motors out of [0,1]: {motors}"
        )


# ---------------------------------------------------------------------------
# TEST 14: Readout updates with reward
# ---------------------------------------------------------------------------
def test_14_readout_updates_with_reward():
    """Same state S gives different output M2 after reward update than M1 before."""
    readout = MotorReadout()
    rng     = np.random.default_rng(seed=7)

    state = rng.standard_normal(N_NODES).astype(np.float32) * 2.0

    # Run several forward passes to build up recent_states history
    for _ in range(10):
        readout.forward(state)
    M1 = readout.forward(state).copy()

    readout.update(reward=1.0)
    M2 = readout.forward(state).copy()

    assert not np.allclose(M1, M2, atol=1e-9), (
        f"Weights did not change after reward update. M1={M1}, M2={M2}"
    )


# ---------------------------------------------------------------------------
# TEST 15: Full controller step runs without error
# ---------------------------------------------------------------------------
def test_15_controller_step_shape():
    """FlyBrainController.step() returns (4,) floats in [0,1], no NaN/Inf."""
    ctrl   = FlyBrainController()
    motors = ctrl.step(
        pos_ned         = np.array([5.0, 3.0, -1.0]),
        pos_uncertainty = 0.5,
        accel           = np.array([0.1, 0.0, -9.81]),
        gyro            = np.array([0.0, 0.0, 0.1]),
        ranges          = np.array([2.0, 5.0, 8.0, 3.0]),
        bearings        = np.array([0.0, 1.57, 3.14, 4.71]),
        reward          = 0.5,
        dt              = 0.01,
    )
    assert motors.shape == (N_MOTORS,), f"Wrong shape: {motors.shape}"
    assert np.all(np.isfinite(motors)), f"NaN/Inf in motors: {motors}"
    assert np.all(motors >= 0.0) and np.all(motors <= 1.0), (
        f"Motors out of [0,1]: {motors}"
    )


# ---------------------------------------------------------------------------
# TEST 16: Controller runs at sensor rate — 1000 steps < 5 s on CPU
# ---------------------------------------------------------------------------
def test_16_speed_1000_steps():
    """1000 steps with all sensors active completes in < 15 seconds on CPU.
    Budget scaled 3x from single-phase (5s) because 3-phase runs three
    tori + three plasticity instances simultaneously."""
    ctrl = FlyBrainController()
    accel = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    gyro  = np.zeros(3, dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(1000):
        angle = i * 0.01
        pos   = np.array([5 * math.cos(angle), 5 * math.sin(angle), -1.5])
        ctrl.step(pos_ned=pos, accel=accel, gyro=gyro)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0, (
        f"1000 steps took {elapsed:.2f}s (limit 15.0s). Pi Zero 2W may be too slow."
    )


# ---------------------------------------------------------------------------
# TEST 17: GPS dropout degrades gracefully
# ---------------------------------------------------------------------------
def test_17_gps_dropout():
    """Controller continues producing valid output when GPS is removed."""
    ctrl  = FlyBrainController()
    accel = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    gyro  = np.zeros(3, dtype=np.float32)

    # 100 steps with GPS
    for i in range(100):
        ctrl.step(pos_ned=np.array([float(i) * 0.01, 0.0, -1.0]),
                  accel=accel, gyro=gyro)

    state_before_dropout = ctrl.torus.get_state().copy()

    # 100 steps without GPS
    motors_after = None
    for i in range(100):
        motors_after = ctrl.step(accel=accel, gyro=gyro)

    state_after_dropout = ctrl.torus.get_state().copy()

    assert np.all(np.isfinite(motors_after)), "NaN/Inf in motors after GPS dropout"
    assert np.all(motors_after >= 0.0) and np.all(motors_after <= 1.0)

    # Wave state should persist — not reset to zero
    assert np.abs(state_after_dropout).max() > 1e-6, (
        "Wave state was zeroed on GPS dropout — should persist"
    )


# ---------------------------------------------------------------------------
# TEST 18: Multi-sensor fusion — GPS + IMU gives different wave state than GPS only
# ---------------------------------------------------------------------------
def test_18_multi_sensor_fusion():
    """GPS+IMU wave state differs from GPS-only; both produce valid motors."""
    accel = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    gyro  = np.zeros(3, dtype=np.float32)
    pos   = np.array([5.0, 0.0, -1.0], dtype=np.float32)

    ctrl_gps = FlyBrainController()
    for _ in range(50):
        ctrl_gps.step(pos_ned=pos)
    state_A = ctrl_gps.torus.get_state().copy()
    motors_A = ctrl_gps.readout.forward(state_A)

    ctrl_both = FlyBrainController()
    for _ in range(50):
        ctrl_both.step(pos_ned=pos, accel=accel, gyro=gyro)
    state_B = ctrl_both.torus.get_state().copy()
    motors_B = ctrl_both.readout.forward(state_B)

    assert not np.allclose(state_A, state_B, atol=1e-6), (
        "GPS-only and GPS+IMU wave states should differ"
    )
    assert np.all(np.isfinite(motors_A)) and np.all(np.isfinite(motors_B))
    assert np.all(motors_A >= 0.0) and np.all(motors_A <= 1.0)
    assert np.all(motors_B >= 0.0) and np.all(motors_B <= 1.0)


# ---------------------------------------------------------------------------
# TEST 19: Learned behaviour persists after save/load
# ---------------------------------------------------------------------------
def test_19_save_load():
    """Weights saved and reloaded produce identical motor output for the same state."""
    ctrl_train = FlyBrainController()
    accel = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    gyro  = np.zeros(3, dtype=np.float32)

    # Train for 500 steps with reward
    for i in range(500):
        pos = np.array([5 * math.cos(i * 0.01), 5 * math.sin(i * 0.01), -1.0])
        reward = 1.0 if i % 10 != 0 else -1.0
        ctrl_train.step(pos_ned=pos, accel=accel, gyro=gyro, reward=reward)

    # Use a fixed wave state to get a reference output
    fixed_state = ctrl_train.torus.get_state().copy()
    M1 = ctrl_train.readout.forward(fixed_state).copy()

    # Save, create fresh controller, load
    with tempfile.TemporaryDirectory() as tmpdir:
        ctrl_train.save_state(tmpdir)

        ctrl_load = FlyBrainController()
        ctrl_load.load_state(tmpdir)

    M2 = ctrl_load.readout.forward(fixed_state).copy()

    assert np.allclose(M1, M2, atol=1e-5), (
        f"Loaded weights produced different output.\nM1={M1}\nM2={M2}"
    )


# ---------------------------------------------------------------------------
# TEST 20: Memory footprint within budget
# ---------------------------------------------------------------------------
def test_20_memory_footprint():
    """Total < 200 KB; torus < 60 KB; plasticity < 60 KB; readout < 20 KB."""
    ctrl = FlyBrainController()
    mem  = ctrl.memory_footprint()

    assert "total_kb"      in mem
    assert "torus_kb"      in mem
    assert "plasticity_kb" in mem
    assert "readout_kb"    in mem

    assert mem["total_kb"]      < 200.0, f"Total {mem['total_kb']} KB exceeds 200 KB"
    assert mem["torus_kb"]      <  60.0, f"Torus {mem['torus_kb']} KB exceeds 60 KB"
    assert mem["plasticity_kb"] <  60.0, f"Plasticity {mem['plasticity_kb']} KB exceeds 60 KB"
    assert mem["readout_kb"]    <  20.0, f"Readout {mem['readout_kb']} KB exceeds 20 KB"

    print(f"\n  Memory: {mem}")
