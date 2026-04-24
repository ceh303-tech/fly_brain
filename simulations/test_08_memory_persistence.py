# fly_brain/simulations/test_08_memory_persistence.py
"""
Test 8: Wave State Memory Duration

Inject obstacle signal for 100 steps.
Remove injection.
Measure how many steps until motor output returns to pre-injection baseline.

This is the wave brain's "working memory duration".
Biologically: a fly's motion detector stays sensitised ~100ms after stimulus.

Pass criterion: torus mean(|V|) remains above baseline+10% for > 5 steps after removal.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from fly_brain.config import V_CRIT
from .sim_environment import CircularEnvironment, RewardCalculator

INJECTION_STEPS  = 100
MAX_DECAY_STEPS  = 200
DECAY_THRESHOLD  = 0.05   # Within this margin = "returned to baseline"
BASELINE_STEPS   = 200    # Steps to establish baseline
OBSTACLE_RANGE   = 0.1    # metres — close obstacle to create strong echo
WARMUP_STEPS     = 2000   # pre-train readout before memory measurement
PAUSE_EVERY      = 200    # warm-up: run free oscillation windows periodically
PAUSE_STEPS      = 10


def main() -> dict:
    """Measure how long wave-based memory persists after stimulus removed."""
    print("\n" + "=" * 60)
    print("Test 8: Wave State Memory Duration")
    print("=" * 60)
    print(f"  Baseline:   {BASELINE_STEPS} steps (no injection)")
    print(f"  Injection:  {INJECTION_STEPS} steps (obstacle at {OBSTACLE_RANGE}m)")
    print(f"  Decay:      up to {MAX_DECAY_STEPS} steps after removal")
    print(f"  Pass:       torus mean(|V|) stays above baseline+10% for > 5 steps")

    ctrl = FlyBrainController()

    # ---- Warm-up: train readout so motor can observe wave memory ----
    print(f"\n  Warm-up: {WARMUP_STEPS} training steps before memory probe...")
    warmup_env = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()
    cycle = PAUSE_EVERY + PAUSE_STEPS
    for i in range(WARMUP_STEPS):
        obs = warmup_env.step()
        reward = reward_calc.compute(obs['min_range'])
        pause_injection = (i % cycle) >= PAUSE_EVERY
        ctrl.step(
            pos_ned=obs['pos_ned'],
            pos_uncertainty=obs['pos_uncertainty'],
            accel=obs['accel'],
            gyro=obs['gyro'],
            ranges=obs['ranges'],
            bearings=obs['bearings'],
            reward=reward,
            dt=0.01,
            inject_sensors=(not pause_injection),
        )
    warmup_env.reset()

    # ---- Phase 1: Establish baseline ----
    print("\n  Phase 1: Establishing baseline...")
    # Run torus for a few warm-up steps first (no sensor inputs)
    for _ in range(50):
        ctrl.step(reward=0.0)

    baseline_motors = []
    baseline_vabs_series = []
    baseline_iabs_series = []
    baseline_spike_series = []
    dummy_ranges    = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    dummy_bearings  = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float32)

    for _ in range(BASELINE_STEPS):
        motors = ctrl.step(
            ranges=dummy_ranges,
            bearings=dummy_bearings,
            reward=0.0,
        )
        baseline_motors.append(float(np.mean(motors)))
        baseline_vabs_series.append(float(np.mean(np.abs(ctrl.torus.V))))
        baseline_iabs_series.append(float(np.mean(np.abs(ctrl.torus.I))))
        baseline_spike_series.append(float(ctrl.torus.spike_rate()))

    baseline_mean = float(np.mean(baseline_motors))
    baseline_std = float(np.std(baseline_motors))
    baseline_vabs = float(np.mean(baseline_vabs_series))
    baseline_iabs = float(np.mean(baseline_iabs_series))
    baseline_spike = float(np.mean(baseline_spike_series))
    print(f"    Baseline motor: mean={baseline_mean:.4f}  std={baseline_std:.4f}")

    # ---- Phase 2: Inject strong obstacle signal ----
    print(f"\n  Phase 2: Injecting obstacle at {OBSTACLE_RANGE}m for {INJECTION_STEPS} steps...")
    close_ranges   = np.array([OBSTACLE_RANGE] * 4, dtype=np.float32)
    close_bearings = dummy_bearings.copy()

    elevated_motors = []
    for _ in range(INJECTION_STEPS):
        motors = ctrl.step(
            ranges   = close_ranges,
            bearings = close_bearings,
            reward   = 0.0,   # no learning — purely observing memory
        )
        elevated_motors.append(float(np.mean(motors)))

    elevated_mean = float(np.mean(elevated_motors))
    print(f"    Elevated motor: mean={elevated_mean:.4f}")

    # If there's no elevation, the test is uninformative
    elevation = elevated_mean - baseline_mean
    print(f"    Elevation above baseline: {elevation:+.4f}")

    # ---- Phase 3: Remove injection and measure decay ----
    print(f"\n  Phase 3: Removing injection, measuring decay...")
    decay_motors   = []
    decay_vabs     = []
    decay_iabs     = []
    decay_spikes   = []
    returned_step  = None

    for step in range(MAX_DECAY_STEPS):
        motors = ctrl.step(
            ranges   = dummy_ranges,
            bearings = dummy_bearings,
            reward   = 0.0,
        )
        m = float(np.mean(motors))
        vabs = float(np.mean(np.abs(ctrl.torus.V)))
        iabs = float(np.mean(np.abs(ctrl.torus.I)))
        spike_rate = float(ctrl.torus.spike_rate())
        decay_motors.append(m)
        decay_vabs.append(vabs)
        decay_iabs.append(iabs)
        decay_spikes.append(spike_rate)

        # Track first return (for display)
        if abs(m - baseline_mean) <= DECAY_THRESHOLD and returned_step is None:
            returned_step = step + 1

    # Primary memory metric: torus-state persistence.
    # Count steps until mean(|V|) returns within 10% of pre-injection baseline.
    v_threshold = max(abs(baseline_vabs) * 0.10, 1e-6)
    first_v_return = next(
        (i + 1 for i, v in enumerate(decay_vabs)
         if abs(v - baseline_vabs) <= v_threshold),
        None,
    )
    if first_v_return is None:
        memory_duration = MAX_DECAY_STEPS
    else:
        memory_duration = max(0, first_v_return - 1)

    # Keep motor persistence as secondary diagnostic only.
    last_elevated = max(
        (i + 1 for i, m in enumerate(decay_motors)
         if abs(m - baseline_mean) > DECAY_THRESHOLD),
        default=0,
    )

    passed = memory_duration > 5

    duration_ms = memory_duration * 10  # each step = 10 ms at 100 Hz

    print(f"\n  Results:")
    print(f"    Baseline motor output:    {baseline_mean:.4f}")
    print(f"    Elevated motor output:    {elevated_mean:.4f}")
    print(f"    Baseline mean(|V|):       {baseline_vabs:.4f}")
    print(f"    First return to baseline at step:    "
          f"{returned_step if returned_step else f'> {MAX_DECAY_STEPS}'}")
    print(f"    First |V| return to baseline±10%: "
          f"{first_v_return if first_v_return is not None else f'> {MAX_DECAY_STEPS}'}")
    print(f"    Last motor-elevated step (diagnostic): {last_elevated}")
    print(f"    Memory duration:          {memory_duration} steps")
    print(f"    Memory duration:          ~{duration_ms} ms at 100 Hz")
    print(f"    Threshold:                > 5 steps (oscillatory wave persistence)")

    # Print torus-state decay diagnostics until all channels return to baseline
    # or 50 steps elapse, whichever comes first.
    v_tol = max(1e-4, abs(baseline_vabs) * 0.05)
    i_tol = max(1e-4, abs(baseline_iabs) * 0.05)
    s_tol = max(1e-4, abs(baseline_spike) * 0.05)
    m_tol = DECAY_THRESHOLD

    settle_step = None
    for i in range(len(decay_motors)):
        if (abs(decay_vabs[i] - baseline_vabs) <= v_tol and
            abs(decay_iabs[i] - baseline_iabs) <= i_tol and
            abs(decay_spikes[i] - baseline_spike) <= s_tol and
            abs(decay_motors[i] - baseline_mean) <= m_tol):
            settle_step = i + 1
            break

    diag_steps = 50 if settle_step is None else min(50, settle_step)
    print(f"\n  Torus-state decay trace (step-by-step, up to {diag_steps} steps):")
    print("  Step | mean(|V|) | mean(|I|) | spike_rate | motor")
    print("  " + "-" * 58)
    for i in range(diag_steps):
        print(f"  {i+1:4d} | {decay_vabs[i]:9.5f} | {decay_iabs[i]:9.5f} |"
              f" {decay_spikes[i]:10.5f} | {decay_motors[i]:.5f}")

    # Print decay trace (first 30 steps)
    n_show = min(30, len(decay_motors))
    print(f"\n  Decay trace (first {n_show} steps after removal):")
    print(f"  Step | Motor | Δ from baseline")
    print(f"  " + "-" * 30)
    for i, m in enumerate(decay_motors[:n_show]):
        delta = m - baseline_mean
        bar_char = "▓" if abs(delta) > DECAY_THRESHOLD else "░"
        bar = bar_char * min(20, int(abs(delta) * 40))
        print(f"  {i+1:4d} | {m:.4f} | {delta:+.4f} {bar}")

    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"memory={memory_duration} steps (~{duration_ms} ms)")

    return {
        'passed':          passed,
        'baseline_mean':   baseline_mean,
        'elevated_mean':   elevated_mean,
        'memory_duration': memory_duration,
        'duration_ms':     duration_ms,
        'metric_label':    f"{memory_duration} steps (~{duration_ms}ms)",
    }


if __name__ == '__main__':
    main()
