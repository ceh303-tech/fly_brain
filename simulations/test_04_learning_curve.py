# fly_brain/simulations/test_04_learning_curve.py
"""
Test 4: Learning Speed

Measure separation at checkpoints throughout training.
Identifies when controller first becomes useful.

Key question: how many steps until separation > 0.2 consistently?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from .sim_environment import (CircularEnvironment, RewardCalculator)

TOTAL_STEPS      = 10000
CHECKPOINTS      = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000]
USEFUL_THRESHOLD = 0.2   # Separation at which controller is "useful"


def _ascii_curve(x_vals, y_vals, width=60, height=10,
                 threshold=None) -> str:
    """Render a simple ASCII line chart. Returns multi-line string."""
    if not y_vals:
        return "  (no data)"

    y_min = 0.0
    y_max = max(max(y_vals) + 0.05, threshold + 0.05 if threshold else 0.55)
    x_min = x_vals[0]
    x_max = x_vals[-1]

    grid = [[' '] * width for _ in range(height)]

    # Plot threshold line
    if threshold is not None:
        ty = int((1.0 - (threshold - y_min) / (y_max - y_min)) * (height - 1))
        ty = max(0, min(height - 1, ty))
        for col in range(width):
            if grid[ty][col] == ' ':
                grid[ty][col] = '-'

    # Plot data points
    for x, y in zip(x_vals, y_vals):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = int((1.0 - (y - y_min) / (y_max - y_min)) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = '*'

    lines = []
    n_ticks = 5
    for r, row in enumerate(grid):
        v = y_max - r * (y_max - y_min) / (height - 1)
        label = f"{v:5.2f} |"
        if r == int((1.0 - (threshold - y_min) / (y_max - y_min)) * (height - 1)) \
                and threshold is not None:
            label = f"{v:5.2f} |"
            line_suffix = "  \u2190 useful threshold" if r == int(
                (1.0 - (threshold - y_min) / (y_max - y_min)) * (height - 1)
            ) else ""
        else:
            line_suffix = ""
        lines.append(label + ''.join(row) + line_suffix)

    # X-axis
    lines.append("       " + "-" * width)
    tick_positions = np.linspace(x_min, x_max, n_ticks)
    tick_labels = [f"{v/1000:.0f}k" if v >= 1000 else str(int(v))
                   for v in tick_positions]
    x_axis = "       "
    for i, label in enumerate(tick_labels):
        pos = int((tick_positions[i] - x_min) / (x_max - x_min) * (width - 1))
        pad = pos - len(x_axis) + 7
        x_axis += " " * max(pad, 0) + label
    lines.append(x_axis)

    return "\n".join(lines)


def main() -> dict:
    """
    Run training from scratch, collecting separation at each checkpoint.
    Returns result dict.
    """
    print("\n" + "=" * 60)
    print("Test 4: Learning Speed / Learning Curve")
    print("=" * 60)
    print(f"  Total steps: {TOTAL_STEPS}")
    print(f"  Useful threshold: separation > {USEFUL_THRESHOLD}")

    ctrl   = FlyBrainController()
    circle = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()

    # Collect per-step data, sample at checkpoints
    checkpoint_set = set(CHECKPOINTS)
    checkpoint_seps = {}

    pos_motors = []
    neg_motors = []
    prev_ckpt  = 0

    for step in range(1, TOTAL_STEPS + 1):
        obs    = circle.step()
        reward = reward_calc.compute(obs['min_range'])

        motors = ctrl.step(
            pos_ned         = obs['pos_ned'],
            pos_uncertainty = obs['pos_uncertainty'],
            accel           = obs['accel'],
            gyro            = obs['gyro'],
            ranges          = obs['ranges'],
            bearings        = obs['bearings'],
            reward          = reward,
        )
        m = float(np.mean(motors))
        if reward > 0:
            pos_motors.append(m)
        else:
            neg_motors.append(m)

        if step in checkpoint_set:
            if pos_motors and neg_motors:
                sep = abs(np.mean(pos_motors) - np.mean(neg_motors))
            else:
                sep = 0.0
            checkpoint_seps[step] = sep
            pos_motors = []
            neg_motors = []
            prev_ckpt = step

    # Find first useful step
    first_useful = None
    for ckpt in CHECKPOINTS:
        if checkpoint_seps.get(ckpt, 0.0) >= USEFUL_THRESHOLD:
            if first_useful is None:
                first_useful = ckpt

    # Print table
    print(f"\n  Learning Curve (separation vs steps):")
    print(f"  {'Steps':>7}  {'Separation':>10}  {'Useful?'}")
    print("  " + "-" * 35)
    for ckpt in CHECKPOINTS:
        sep = checkpoint_seps.get(ckpt, 0.0)
        useful = "✓" if sep >= USEFUL_THRESHOLD else ""
        print(f"  {ckpt:>7}  {sep:10.4f}  {useful}")

    # ASCII plot
    x_vals = CHECKPOINTS
    y_vals = [checkpoint_seps.get(c, 0.0) for c in CHECKPOINTS]
    print(f"\n  ASCII Learning Curve (separation vs steps):")
    print(_ascii_curve(x_vals, y_vals, threshold=USEFUL_THRESHOLD))

    if first_useful is not None:
        print(f"\n  Controller first useful at step {first_useful}")
    else:
        print(f"\n  Controller did not reach useful threshold "
              f"({USEFUL_THRESHOLD}) in {TOTAL_STEPS} steps")

    final_sep = checkpoint_seps.get(TOTAL_STEPS, 0.0)
    passed = (first_useful is not None)

    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"useful_step={first_useful}  final_sep={final_sep:.3f}")

    return {
        'passed':           passed,
        'first_useful_step': first_useful,
        'final_separation': final_sep,
        'checkpoint_seps':  checkpoint_seps,
        'metric_label':     f"useful@{first_useful}" if first_useful else "not_useful",
    }


if __name__ == '__main__':
    main()
