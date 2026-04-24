# fly_brain/simulations/test_01_generalisation.py
"""
Test 1: Generalisation — IMU Noise Robustness

Train on circular orbit with rng_seed=42 (specific IMU noise realisation).
Freeze weights. Evaluate on same orbit geometry with rng_seed=99
(independent IMU noise never seen during training).

Obstacle positions are purely geometric (cardinal-point linspace) and
are IDENTICAL across the two seeds.  Only the inertial sensor noise
(accel ±0.05 m/s², gyro ±0.01 rad/s) differs between the two instances.

If the controller relied on memorised IMU noise patterns the separation
would collapse on the novel noise seed.  Robustness demonstrates the
readout learned from ultrasonic range signals, not coincident IMU noise.

Pass criterion: frozen-eval separation > 0.10.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from .sim_environment import (CircularEnvironment, TrainingRunner, RewardCalculator)

PASS_THRESHOLD   = 0.10
TRAIN_STEPS      = 5000
TEST_STEPS       = 1000
WEIGHTS_DIR      = os.path.join(os.path.dirname(__file__),
                                 '..', 'results', 'weights_post_circle')

TRAIN_RNG_SEED   = 42   # IMU noise during training
EVAL_RNG_SEED    = 99   # Independent IMU noise — never seen during training


def main() -> dict:
    """
    Train then evaluate with independent IMU noise.
    Returns result dict.
    """
    np.random.seed(10)   # deterministic torus thermal noise
    print("\n" + "=" * 60)
    print("Test 1: Generalisation — IMU Noise Robustness")
    print("=" * 60)
    print(f"  Train: {TRAIN_STEPS} steps — circle, IMU noise seed {TRAIN_RNG_SEED}")
    print(f"  Test:  {TEST_STEPS} steps — same orbit geometry, IMU noise seed {EVAL_RNG_SEED}")
    print(f"  Pass:  frozen-eval separation > {PASS_THRESHOLD}")
    print(f"  Claim: separation persists under unseen IMU noise realisation")

    ctrl      = FlyBrainController()
    train_env = CircularEnvironment(rng_seed=TRAIN_RNG_SEED)

    # ---- Phase 1: Train ----
    print("\n  Training...")
    metrics = TrainingRunner.run(ctrl, train_env, TRAIN_STEPS, collect_every=500)
    sep_train = metrics['final_separation']
    print(f"  Training separation: {sep_train:.3f}")

    # Save weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    ctrl.save_state(WEIGHTS_DIR)
    print(f"  Weights saved to: {WEIGHTS_DIR}")

    # ---- Phase 2: Frozen eval with novel IMU noise seed ----
    print(f"\n  Evaluating with noise seed {EVAL_RNG_SEED} (independent of training)...")
    eval_env    = CircularEnvironment(rng_seed=EVAL_RNG_SEED)
    reward_calc = RewardCalculator()

    pos_motors = []
    neg_motors = []

    for _ in range(TEST_STEPS):
        obs    = eval_env.step()
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
        sep_eval = abs(np.mean(pos_motors) - np.mean(neg_motors))
    else:
        sep_eval = 0.0

    passed = sep_eval > PASS_THRESHOLD

    print(f"\n  Results:")
    print(f"    Separation (training  seed {TRAIN_RNG_SEED}): {sep_train:.3f}")
    print(f"    Separation (eval seed {EVAL_RNG_SEED}, frozen):  {sep_eval:.3f}")
    print(f"    Threshold:                             {PASS_THRESHOLD}")
    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"sep_eval={sep_eval:.3f}  "
          f"({'>' if passed else '<='} {PASS_THRESHOLD})")
    if passed:
        print(f"  Readout response is robust to unseen IMU noise — "
              f"learned from range signals, not noise artefacts.")

    return {
        'passed':           passed,
        'separation_train': sep_train,
        'separation_eval':  sep_eval,
        'metric_label':     f"sep={sep_eval:.3f}",
    }


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
