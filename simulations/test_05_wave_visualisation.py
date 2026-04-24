# fly_brain/simulations/test_05_wave_visualisation.py
"""
Test 5: Wave State Visualisation Over Training

Capture ASCII torus snapshots at training checkpoints.
Shows learning as spatial structure emerging in wave geometry.

Not a pass/fail test — generates visual evidence for the paper.
Saves snapshots to fly_brain/results/wave_snapshots.txt
"""

import sys
import os
import io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from fly_brain.config import TORUS_W, TORUS_H, V_CRIT
from .sim_environment import CircularEnvironment, RewardCalculator

TOTAL_STEPS     = 10000
SNAPSHOT_STEPS  = [0, 1000, 3000, 5000, 10000]
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_FILE     = os.path.join(RESULTS_DIR, 'wave_snapshots.txt')

# Sensor region boundaries (x0, y0, x1, y1) from config
GPS_X0,   GPS_Y0,   GPS_X1,   GPS_Y1   = 0,  0,  16, 16
IMU_X0,   IMU_Y0,   IMU_X1,   IMU_Y1   = 16, 0,  32, 16
SONIC_X0, SONIC_Y0, SONIC_X1, SONIC_Y1 = 0,  16, 16, 32
FLOW_X0,  FLOW_Y0,  FLOW_X1,  FLOW_Y1  = 16, 16, 32, 32


def _render_snapshot(ctrl, step: int, spike_rate: float) -> str:
    """
    Render torus voltage as 32×32 ASCII grid with region labels.
    Returns a string ready to print or save.
    """
    state = ctrl.torus.get_state().reshape(TORUS_H, TORUS_W)
    lines = []
    lines.append(f"\nStep {step:>6}  (spike rate: {spike_rate*100:.1f}%)")

    # Region boundary header
    half = TORUS_W // 2
    lines.append("+" + "-" * half + "+" + "-" * half + "+")
    lines.append(
        "|" + " GPS REGION ".center(half) +
        "|" + " IMU REGION ".center(half) + "|"
    )

    # Torus rows
    for row in range(TORUS_H):
        line = "|"
        for col in range(TORUS_W):
            v = float(state[row, col])
            if   v >= V_CRIT * 1.5: line += "#"
            elif v >= V_CRIT * 0.5: line += "+"
            elif v >= 0.1:          line += "."
            else:                   line += " "
            # Vertical boundary
            if col == half - 1:
                line += "|"
        line += "|"
        # Insert horizontal region separator between row 15 and 16
        if row == TORUS_H // 2 - 1:
            lines.append(line)
            lines.append("+" + "-" * half + "+" + "-" * half + "+")
            lines.append(
                "|" + " SONIC REGION ".center(half) +
                "|" + " FLOW REGION ".center(half) + "|"
            )
        else:
            lines.append(line)

    lines.append("+" + "-" * half + "+" + "-" * half + "+")
    lines.append("Legend: '#' strongly active  '+' moderate  '.' weak  ' ' silent")
    return "\n".join(lines)


def main() -> dict:
    """
    Run training, capturing and saving torus snapshots at checkpoints.
    Returns result dict (always passes — this is a visualisation test).
    """
    print("\n" + "=" * 60)
    print("Test 5: Wave State Visualisation")
    print("=" * 60)
    print(f"  Snapshots at steps: {SNAPSHOT_STEPS}")
    print(f"  Output:             {OUTPUT_FILE}")

    ctrl        = FlyBrainController()
    circle      = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()

    snapshot_set = set(SNAPSHOT_STEPS)
    snapshots    = []
    output_lines = [
        "fly_brain Wave State Snapshots",
        "=" * 70,
        "Training environment: circular orbit with 4 cardinal obstacles",
        "Profile: NORMAL  |  Grid: 32x32  |  DT: 0.01s",
        "=" * 70,
    ]

    # Snapshot at step 0 (before any training)
    if 0 in snapshot_set:
        snap = _render_snapshot(ctrl, 0, ctrl.torus.spike_rate())
        print(snap)
        snapshots.append(snap)
        output_lines.append(snap)

    for step in range(1, TOTAL_STEPS + 1):
        obs    = circle.step()
        reward = reward_calc.compute(obs['min_range'])
        ctrl.step(
            pos_ned         = obs['pos_ned'],
            pos_uncertainty = obs['pos_uncertainty'],
            accel           = obs['accel'],
            gyro            = obs['gyro'],
            ranges          = obs['ranges'],
            bearings        = obs['bearings'],
            reward          = reward,
        )

        if step in snapshot_set and step > 0:
            snap = _render_snapshot(ctrl, step, ctrl.torus.spike_rate())
            print(snap)
            snapshots.append(snap)
            output_lines.append(snap)

    # Save to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\n  Snapshots saved to: {OUTPUT_FILE}")
    print(f"\n  INFO  (no pass/fail — visual evidence for paper)")

    return {
        'passed':       True,   # visualisation test always "passes"
        'n_snapshots':  len(snapshots),
        'output_file':  OUTPUT_FILE,
        'metric_label': f"{len(snapshots)} snapshots saved",
        'is_info_only': True,
    }


if __name__ == '__main__':
    main()
