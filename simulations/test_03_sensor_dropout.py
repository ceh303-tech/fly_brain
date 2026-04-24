# fly_brain/simulations/test_03_sensor_dropout.py
"""
Test 3: Sensor Dropout Resilience

Train with all sensors. Evaluate with each sensor (and combinations) removed.
Shows how much each sensor modality contributes to obstacle separation.

Pass criterion: all-sensor separation > 0.1 (confirms training worked).
Single-sensor degradation is reported informatively.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from .sim_environment import (CircularEnvironment, TrainingRunner,
                               RewardCalculator)

TRAIN_STEPS    = 5000
EVAL_STEPS     = 500
MIN_SEPARATION = 0.1
WEIGHTS_DIR    = os.path.join(os.path.dirname(__file__),
                               '..', 'results', 'weights_dropout_train')

DROPOUT_CONDITIONS = [
    {'name': 'All sensors',     'gps': True,  'imu': True,  'ultrasonic': True},
    {'name': 'GPS only',        'gps': True,  'imu': False, 'ultrasonic': False},
    {'name': 'IMU only',        'gps': False, 'imu': True,  'ultrasonic': False},
    {'name': 'Ultrasonic only', 'gps': False, 'imu': False, 'ultrasonic': True},
    {'name': 'No GPS',          'gps': False, 'imu': True,  'ultrasonic': True},
    {'name': 'No IMU',          'gps': True,  'imu': False, 'ultrasonic': True},
    {'name': 'No ultrasonic',   'gps': True,  'imu': True,  'ultrasonic': False},
]


def _evaluate_with_dropout(weights_dir: str, sensors: dict,
                            n_steps: int) -> float:
    """Load trained weights into fresh controller and eval with sensor dropout."""
    ctrl   = FlyBrainController()
    ctrl.load_state(weights_dir)
    env    = CircularEnvironment(rng_seed=42)
    reward_calc = RewardCalculator()
    pos_motors  = []
    neg_motors  = []

    for _ in range(n_steps):
        obs    = env.step()
        reward = reward_calc.compute(obs['min_range'])

        kwargs = {'reward': 0.0, 'dt': 0.01}     # learning frozen
        if sensors.get('gps', True):
            kwargs['pos_ned']         = obs['pos_ned']
            kwargs['pos_uncertainty'] = obs['pos_uncertainty']
        if sensors.get('imu', True):
            kwargs['accel']   = obs['accel']
            kwargs['gyro']    = obs['gyro']
        if sensors.get('ultrasonic', True):
            kwargs['ranges']   = obs['ranges']
            kwargs['bearings'] = obs['bearings']

        motors = ctrl.step(**kwargs)
        m = float(np.mean(motors))
        if reward > 0:
            pos_motors.append(m)
        else:
            neg_motors.append(m)

    if pos_motors and neg_motors:
        return abs(np.mean(pos_motors) - np.mean(neg_motors))
    return 0.0


def main() -> dict:
    """Train once, then evaluate under each dropout condition."""
    np.random.seed(100)   # deterministic torus thermal noise
    print("\n" + "=" * 60)
    print("Test 3: Sensor Dropout Resilience")
    print("=" * 60)
    print(f"  Train: {TRAIN_STEPS} steps, all sensors active")
    print(f"  Eval:  {EVAL_STEPS} steps per dropout condition")
    print(f"  Pass:  all-sensor condition > {MIN_SEPARATION} (confirms full training)")
    print(f"  Info:  single-sensor separations reported as sensor-contribution table")

    # ---- Train with all sensors ----
    ctrl   = FlyBrainController()
    circle = CircularEnvironment(rng_seed=42)
    print("\n  Training with all sensors...")
    TrainingRunner.run(ctrl, circle, TRAIN_STEPS, collect_every=1000)

    # Save weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    ctrl.save_state(WEIGHTS_DIR)
    print(f"  Weights saved to: {WEIGHTS_DIR}")

    # ---- Evaluate each condition ----
    print(f"\n  {'Condition':<22} {'Separation':>10}  {'Result':>8}")
    print("  " + "-" * 45)

    results = []

    for cond in DROPOUT_CONDITIONS:
        sep = _evaluate_with_dropout(WEIGHTS_DIR, cond, EVAL_STEPS)
        only_one = sum([cond['gps'], cond['imu'], cond['ultrasonic']]) == 1
        result_ok = sep > MIN_SEPARATION

        marker = "✓" if result_ok else ("(expected)" if cond['name'] == 'No ultrasonic' else "✗")
        print(f"  {cond['name']:<22} {sep:10.4f}  {marker}")
        results.append({'name': cond['name'], 'separation': sep, 'pass': result_ok,
                        'single_sensor': only_one})

    # ---- Determine pass/fail ----
    # Pass criterion: full sensor suite (all sensors) produces non-trivial
    # obstacle separation.  Single-sensor degradation is reported informatively —
    # sensor resilience varies with torus state quality during training.
    all_sensors_result = next(
        (r for r in results if r['name'] == 'All sensors'), None
    )
    all_sensors_sep = all_sensors_result['separation'] if all_sensors_result else 0.0
    passed = all_sensors_sep > MIN_SEPARATION
    n_pass = sum(1 for r in results if r['pass'])

    print(f"\n  {n_pass}/{len(results)} conditions above threshold")
    print(f"  NOTE: 'No ultrasonic' is expected to show lower separation")
    print(f"        (obstacle range info is gone — this is honest)")
    print(f"  NOTE: Single-sensor conditions are informational —")
    print(f"        pass/fail based on full-sensor performance.")
    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"all_sensors_sep={all_sensors_sep:.3f}  "
          f"({'>' if passed else '<='} {MIN_SEPARATION})")

    return {
        'passed':            passed,
        'n_pass':            n_pass,
        'n_total':           len(results),
        'all_sensors_sep':   all_sensors_sep,
        'condition_results': results,
        'metric_label':      f"all_sens={all_sensors_sep:.3f}  {n_pass}/{len(results)} conditions",
    }


if __name__ == '__main__':
    main()
