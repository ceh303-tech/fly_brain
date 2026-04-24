# fly_brain/simulations/test_02_catastrophic_forgetting.py
"""
Test 2: Catastrophic Forgetting

Train on circle (5000 steps). Measure separation_A1.
Then train on square (5000 steps). Re-measure separation on circle (A2).
Report degradation both with and without experience replay.

Pass criterion: degradation < 50% WITH replay active
(confirms experience replay preserves earlier learning when switching environments).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from .sim_environment import (CircularEnvironment, SquareEnvironment,
                               TrainingRunner, RewardCalculator)

TRAIN_STEPS_EACH  = 5000
EVAL_STEPS        = 500
# Pass criterion: degradation WITH replay must be below this threshold
REPLAY_DEGRADATION_THRESHOLD = 0.50  # 50%
# Legacy learning threshold (still reported)
LEARNED_THRESHOLD = 0.05


def _evaluate_separation(ctrl, environment, n_steps: int) -> float:
    """Run eval steps (reward=0 to freeze learning) and return separation."""
    environment.reset()
    reward_calc = RewardCalculator()
    pos_motors = []
    neg_motors = []

    for _ in range(n_steps):
        obs    = environment.step()
        reward = reward_calc.compute(obs['min_range'])
        motors = ctrl.step(
            pos_ned         = obs['pos_ned'],
            pos_uncertainty = obs['pos_uncertainty'],
            accel           = obs['accel'],
            gyro            = obs['gyro'],
            ranges          = obs['ranges'],
            bearings        = obs['bearings'],
            reward          = 0.0,   # freeze learning
        )
        m = float(np.mean(motors))
        if reward > 0:
            pos_motors.append(m)
        else:
            neg_motors.append(m)

    if pos_motors and neg_motors:
        return abs(np.mean(pos_motors) - np.mean(neg_motors))
    return 0.0


def _run_forgetting_trial(use_replay: bool, seed: int) -> dict:
    """
    Run a full train-A / eval-A1 / train-B / eval-A2 sequence.
    Returns dict with sep_a1, sep_a2, degradation.
    """
    np.random.seed(seed)
    ctrl   = FlyBrainController(use_replay=use_replay)
    circle = CircularEnvironment(rng_seed=seed)
    square = SquareEnvironment(rng_seed=seed + 57)

    # Phase 1: Train on circle
    TrainingRunner.run(ctrl, circle, TRAIN_STEPS_EACH, collect_every=500)

    # Eval A1: circle separation after circle training
    sep_a1 = _evaluate_separation(ctrl, circle, EVAL_STEPS)

    # Consolidate circle weights as EWC anchor before switching environments.
    # This anchors important circle connections so EWC resists overwriting them.
    ctrl.consolidate_memory()

    # Phase 2: Train on square (same weights, continuing)
    square.reset()
    TrainingRunner.run(ctrl, square, TRAIN_STEPS_EACH, collect_every=500)

    # Eval A2: circle separation after square training
    sep_a2 = _evaluate_separation(ctrl, circle, EVAL_STEPS)

    if sep_a1 > 1e-6:
        degradation = (sep_a1 - sep_a2) / sep_a1
    else:
        degradation = 0.0

    return {
        'sep_a1':      sep_a1,
        'sep_a2':      sep_a2,
        'degradation': degradation,
    }


def main() -> dict:
    """
    Run both with-replay and without-replay trials.
    Pass criterion: degradation < 50% with replay.
    """
    print("\n" + "=" * 60)
    print("Test 2: Catastrophic Forgetting — Experience Replay")
    print("=" * 60)
    print(f"  Phase 1: {TRAIN_STEPS_EACH} steps on circle")
    print(f"  Phase 2: {TRAIN_STEPS_EACH} steps on square")
    print(f"  Phase 3: Evaluate on circle again")
    print(f"  Pass:    degradation < {REPLAY_DEGRADATION_THRESHOLD*100:.0f}% with replay active")

    # ---- Run WITHOUT replay (baseline) ----
    print("\n  Running WITHOUT replay (baseline)...")
    no_replay = _run_forgetting_trial(use_replay=False, seed=42)
    print(f"    sep_a1={no_replay['sep_a1']:.4f}  "
          f"sep_a2={no_replay['sep_a2']:.4f}  "
          f"deg={no_replay['degradation']*100:.1f}%")

    # ---- Run WITH replay ----
    print("\n  Running WITH replay...")
    with_replay = _run_forgetting_trial(use_replay=True, seed=42)
    print(f"    sep_a1={with_replay['sep_a1']:.4f}  "
          f"sep_a2={with_replay['sep_a2']:.4f}  "
          f"deg={with_replay['degradation']*100:.1f}%")

    # ---- Compute improvement ----
    improvement_pp = (no_replay['degradation'] - with_replay['degradation']) * 100

    # ---- Pass criterion ----
    passed = (with_replay['degradation'] < REPLAY_DEGRADATION_THRESHOLD
              and with_replay['sep_a1'] > LEARNED_THRESHOLD)

    print(f"\n  Catastrophic Forgetting Results:")
    print(f"    Without replay: sep_a1={no_replay['sep_a1']:.3f}  "
          f"sep_a2={no_replay['sep_a2']:.3f}  "
          f"deg={no_replay['degradation']*100:.0f}%")
    print(f"    With replay:    sep_a1={with_replay['sep_a1']:.3f}  "
          f"sep_a2={with_replay['sep_a2']:.3f}  "
          f"deg={with_replay['degradation']*100:.0f}%")
    print(f"    Improvement:    {improvement_pp:.1f} percentage points")
    print(f"    Pass criterion: degradation < {REPLAY_DEGRADATION_THRESHOLD*100:.0f}% with replay")

    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"sep_a1={with_replay['sep_a1']:.3f}  "
          f"deg={with_replay['degradation']*100:.0f}%"
          f"  ({'<' if with_replay['degradation'] < REPLAY_DEGRADATION_THRESHOLD else '>='}"
          f" {REPLAY_DEGRADATION_THRESHOLD*100:.0f}%)")

    return {
        'passed':      passed,
        'sep_a1':      with_replay['sep_a1'],
        'sep_a2':      with_replay['sep_a2'],
        'degradation': with_replay['degradation'],
        'no_replay_degradation': no_replay['degradation'],
        'improvement_pp': improvement_pp,
        'metric_label': (f"sep_a1={with_replay['sep_a1']:.3f}  "
                         f"deg={with_replay['degradation']*100:.0f}%"),
    }


if __name__ == '__main__':
    main()

