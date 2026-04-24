# fly_brain/simulations/test_06_thermodynamic_profiles.py
"""
Test 6: Thermodynamic Profile Comparison

Run identical training on each thermodynamic profile.
Identifies which profile learns best for navigation.

Pass criterion: NORMAL profile separation > CHAOTIC profile separation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from fly_brain.config import THERMODYNAMIC_PROFILES
from .sim_environment import (CircularEnvironment, TrainingRunner,
                               RewardCalculator)

TRAIN_STEPS      = 5000
EVAL_STEPS       = 500
PROFILES_TO_TEST = ['FROZEN', 'CALM', 'NORMAL', 'EXCITED', 'CHAOTIC']


def _evaluate_separation(ctrl, n_steps: int) -> float:
    """Return separation on a fresh circle (learning frozen)."""
    env = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()
    pos_motors = []
    neg_motors = []
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


def main() -> dict:
    """Train each profile and compare. Returns result dict."""
    print("\n" + "=" * 60)
    print("Test 6: Thermodynamic Profile Comparison")
    print("=" * 60)
    print(f"  Profiles: {PROFILES_TO_TEST}")
    print(f"  Train: {TRAIN_STEPS} steps  |  Eval: {EVAL_STEPS} steps")
    print(f"  Pass:  NORMAL separation > CHAOTIC separation")

    profile_results = {}

    for profile in PROFILES_TO_TEST:
        p = THERMODYNAMIC_PROFILES[profile]
        print(f"\n  Profile: {profile:8s}  "
              f"R={p['R']:.3f}  L={p['L']:.4f}  "
              f"C={p['C']:.4f}  V_CRIT={p['V_CRIT']:.1f}")

        ctrl   = FlyBrainController(profile=profile)
        circle = CircularEnvironment(rng_seed=42)

        # Train
        metrics = TrainingRunner.run(
            ctrl, circle, TRAIN_STEPS, collect_every=1000
        )
        mean_spike = float(np.mean(metrics['spike_rates']))

        # Evaluate
        sep = _evaluate_separation(ctrl, EVAL_STEPS)
        profile_results[profile] = {
            'separation':  sep,
            'mean_spike':  mean_spike,
            'mean_weight': metrics['mean_weights'][-1] if metrics['mean_weights'] else float('nan'),
        }
        print(f"    separation={sep:.4f}  spike%={mean_spike*100:.1f}%")

    # Summary table
    print(f"\n  Profile Comparison:")
    print(f"  {'Profile':<10} {'Separation':>10}  {'Spike%':>7}  {'MeanW':>7}")
    print("  " + "-" * 40)
    for name in PROFILES_TO_TEST:
        r = profile_results[name]
        indicator = " ← best" if r['separation'] == max(
            v['separation'] for v in profile_results.values()
        ) else ""
        print(f"  {name:<10} {r['separation']:10.4f}  "
              f"{r['mean_spike']*100:6.1f}%  {r['mean_weight']:7.4f}{indicator}")

    normal_sep  = profile_results.get('NORMAL',  {}).get('separation', 0.0)
    chaotic_sep = profile_results.get('CHAOTIC', {}).get('separation', 0.0)
    passed      = normal_sep > chaotic_sep

    print(f"\n  NORMAL separation:  {normal_sep:.4f}")
    print(f"  CHAOTIC separation: {chaotic_sep:.4f}")
    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"NORMAL({normal_sep:.3f}) {'>' if passed else '<='} CHAOTIC({chaotic_sep:.3f})")

    if not passed:
        print("  NOTE: If CHAOTIC >= NORMAL, the high firing rate may produce")
        print("        decorrelated states that accidentally separate — but the")
        print("        controller would not be reliable or interpretable.")

    return {
        'passed':          passed,
        'normal_sep':      normal_sep,
        'chaotic_sep':     chaotic_sep,
        'profile_results': profile_results,
        'metric_label':    f"NORMAL={normal_sep:.3f}>CHAOTIC={chaotic_sep:.3f}"
                           if passed else f"NORMAL={normal_sep:.3f}<=CHAOTIC={chaotic_sep:.3f}",
    }


if __name__ == '__main__':
    main()
