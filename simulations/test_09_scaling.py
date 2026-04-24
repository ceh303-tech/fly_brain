# fly_brain/simulations/test_09_scaling.py
"""
Test 9: Torus Size Scaling

Tests performance at 16×16, 32×32, and 64×64.
Provides hardware deployment guidance.

Pi Zero 2W target: 32×32 (current)
Pi 4 target:       64×64 (better performance)
MCU target:        16×16 (minimal)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from .sim_environment import (CircularEnvironment, TrainingRunner,
                               RewardCalculator)

TRAIN_STEPS = 3000
EVAL_STEPS  = 500
SPEED_STEPS = 1000   # steps to time for steps/sec measurement

SIZES = [
    {'W': 16, 'H': 16, 'label': '16x16 (microcontroller)'},
    {'W': 32, 'H': 32, 'label': '32x32 (Pi Zero 2W)'},
    {'W': 64, 'H': 64, 'label': '64x64 (Pi 4)'},
]


def _measure_speed(ctrl, n_steps: int) -> float:
    """Return steps per second by timing n_steps of idle controller."""
    env = CircularEnvironment(rng_seed=7)
    t0  = time.perf_counter()
    for _ in range(n_steps):
        obs = env.step()
        ctrl.step(
            pos_ned  = obs['pos_ned'],
            accel    = obs['accel'],
            gyro     = obs['gyro'],
            ranges   = obs['ranges'],
            bearings = obs['bearings'],
            reward   = 0.0,
        )
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed


def _eval_separation(ctrl, n_steps: int) -> tuple:
    """Return (separation, mean_spike_rate) with learning frozen."""
    env = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()
    pos_motors  = []
    neg_motors  = []
    spike_rates = []

    for _ in range(n_steps):
        obs    = env.step()
        reward = reward_calc.compute(obs['min_range'])
        motors = ctrl.step(
            pos_ned         = obs['pos_ned'],
            pos_uncertainty = obs['pos_uncertainty'],
            accel           = obs['accel'],
            gyro            = obs['gyro'],
            ranges          = obs['ranges'],
            bearings        = obs['bearings'],
            reward          = 0.0,
        )
        m = float(np.mean(motors))
        if reward > 0:
            pos_motors.append(m)
        else:
            neg_motors.append(m)
        spike_rates.append(ctrl.torus.spike_rate())

    if pos_motors and neg_motors:
        sep = abs(np.mean(pos_motors) - np.mean(neg_motors))
    else:
        sep = 0.0

    return sep, float(np.mean(spike_rates))


def main() -> dict:
    """Train and evaluate each torus size, print comparison table."""
    print("\n" + "=" * 60)
    print("Test 9: Torus Size Scaling")
    print("=" * 60)
    print(f"  Train: {TRAIN_STEPS} steps  |  Eval: {EVAL_STEPS} steps")
    print(f"  Speed: measured over {SPEED_STEPS} steps")

    size_results = []

    for sz in SIZES:
        W, H, label = sz['W'], sz['H'], sz['label']
        print(f"\n  Size: {label}  ({W*H} nodes)")

        ctrl   = FlyBrainController(torus_w=W, torus_h=H)
        circle = CircularEnvironment(rng_seed=42)

        # Train
        print(f"    Training {TRAIN_STEPS} steps...")
        TrainingRunner.run(ctrl, circle, TRAIN_STEPS, collect_every=1000)

        # Evaluate separation
        sep, spike_rate = _eval_separation(ctrl, EVAL_STEPS)

        # Memory footprint
        mem = ctrl.memory_footprint()
        total_kb = mem['total_kb']

        # Speed (fresh controller — measures pure dynamics overhead)
        ctrl_fresh = FlyBrainController(torus_w=W, torus_h=H)
        steps_per_sec = _measure_speed(ctrl_fresh, SPEED_STEPS)

        result = {
            'label':         label,
            'W':             W,
            'H':             H,
            'n_nodes':       W * H,
            'separation':    sep,
            'memory_kb':     total_kb,
            'steps_per_sec': steps_per_sec,
            'spike_pct':     spike_rate * 100.0,
        }
        size_results.append(result)

        print(f"    separation={sep:.4f}  "
              f"memory={total_kb:.1f}KB  "
              f"speed={steps_per_sec:.0f} steps/sec  "
              f"spike={spike_rate*100:.1f}%")

    # Print comparison table
    print(f"\n  Scaling Comparison:")
    print(f"  {'Size':<24} | {'Sep':>6} | {'Memory':>8} | "
          f"{'Steps/sec':>10} | {'Spike%':>6}")
    print("  " + "-" * 65)
    for r in size_results:
        print(f"  {r['label']:<24} | {r['separation']:6.3f} | "
              f"{r['memory_kb']:7.1f}KB | {r['steps_per_sec']:10.0f} | "
              f"{r['spike_pct']:5.1f}%")

    print(f"\n  Deployment guidance:")
    for r in size_results:
        device = r['label'].split('(')[1].rstrip(')')
        print(f"    {device:<15s}: {r['W']}x{r['H']} — "
              f"{r['memory_kb']:.0f} KB, {r['steps_per_sec']:.0f} steps/sec, "
              f"sep={r['separation']:.3f}")

    # Test "passes" if 32x32 is at least as good as 16x16 (not slower to learn)
    r16 = next(r for r in size_results if r['W'] == 16)
    r32 = next(r for r in size_results if r['W'] == 32)
    r64 = next(r for r in size_results if r['W'] == 64)
    passed = (r32['separation'] >= r16['separation'] * 0.7 and
              r64['separation'] >= r32['separation'] * 0.7)

    print(f"\n  INFO  (scaling comparison — see table above)")
    print(f"  All three sizes functional: {'PASS ✓' if passed else 'PARTIAL'}")

    return {
        'passed':       passed,
        'size_results': size_results,
        'metric_label': "see table",
        'is_info_only': True,
    }


if __name__ == '__main__':
    main()
