# fly_brain/simulations/sim_environment.py
"""
Shared simulation environment for fly_brain validation.

Provides:
- CircularEnvironment: drone on circular orbit with obstacles
- SquareEnvironment:   drone on square path with obstacles
- RewardCalculator:    consistent reward signal from sensor state
- TrainingRunner:      run N steps and return metrics
- MetricsCollector:    track separation, learning, memory stats
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fly_brain.controller import FlyBrainController
from fly_brain.config import (TORUS_W, TORUS_H, V_CRIT, N_MOTORS,
                               GPS_REGION, IMU_REGION, SONIC_REGION)


class CircularEnvironment:
    """
    Simulated drone on circular orbit.
    Obstacles at north, south, east, west at radius 3m.
    Drone orbit radius: 5m.
    At orbit speed the drone passes close to each obstacle
    once per revolution.

    Used for: training, baseline, thermodynamic profile tests.
    """

    def __init__(self, radius: float = 5.0,
                 obstacle_range: float = 4.5,
                 n_obstacles: int = 4,
                 rng_seed: int = 42):
        self.radius = radius
        self.obstacle_range = obstacle_range
        self.n_obstacles = n_obstacles
        self.rng = np.random.RandomState(rng_seed)
        self.t = 0.0
        self.dt = 0.01
        self.omega = 0.1  # rad/s orbital angular velocity

        # Obstacle positions (fixed, cardinal points)
        angles = np.linspace(0, 2 * np.pi, n_obstacles, endpoint=False)
        self.obstacles_ned = np.array([
            [obstacle_range * np.cos(a), obstacle_range * np.sin(a)]
            for a in angles
        ])

    def step(self) -> dict:
        """
        Advance environment one timestep.
        Returns sensor readings and ground truth for reward calculation.
        """
        self.t += self.dt

        # Drone position on circular orbit
        pos_n = self.radius * np.cos(self.omega * self.t)
        pos_e = self.radius * np.sin(self.omega * self.t)
        pos_ned = np.array([pos_n, pos_e, -10.0], dtype=np.float32)

        # Velocity (tangential to orbit)
        vel_n = -self.radius * self.omega * np.sin(self.omega * self.t)
        vel_e =  self.radius * self.omega * np.cos(self.omega * self.t)

        # IMU: gravity + centripetal acceleration
        accel = np.array([
            -self.radius * self.omega**2 * np.cos(self.omega * self.t),
            -self.radius * self.omega**2 * np.sin(self.omega * self.t),
            -9.81,
        ], dtype=np.float32)
        accel += self.rng.randn(3).astype(np.float32) * 0.05
        gyro = np.array([0.0, 0.0, self.omega], dtype=np.float32)
        gyro += self.rng.randn(3).astype(np.float32) * 0.01

        # Ultrasonic: range to each obstacle
        drone_pos_2d = np.array([pos_n, pos_e])
        ranges = []
        bearings = []
        for obs in self.obstacles_ned:
            diff = obs - drone_pos_2d
            dist = float(np.linalg.norm(diff))
            bearing = float(np.arctan2(diff[1], diff[0]))
            ranges.append(dist)
            bearings.append(bearing)

        ranges = np.array(ranges, dtype=np.float32)
        bearings = np.array(bearings, dtype=np.float32)
        min_range = float(np.min(ranges))

        return {
            'pos_ned': pos_ned,
            'pos_uncertainty': 0.5,
            'accel': accel,
            'gyro': gyro,
            'ranges': ranges,
            'bearings': bearings,
            'min_range': min_range,
            'vel_ned': np.array([vel_n, vel_e, 0.0], dtype=np.float32),
        }

    def reset(self):
        """Reset to initial state."""
        self.t = 0.0


class SquareEnvironment:
    """
    Simulated drone on square path.
    Obstacles at four corners.
    NEVER used for training — only for generalisation tests.

    Used for: Test 1 (generalisation to novel environment).
    """

    def __init__(self, side: float = 5.0,
                 obstacle_range: float = 2.0,
                 rng_seed: int = 99):
        self.side = side
        self.obstacle_range = obstacle_range
        self.rng = np.random.RandomState(rng_seed)
        self.t = 0.0
        self.dt = 0.01
        self.speed = 0.5  # m/s along path

        # Square waypoints
        h = side / 2.0
        self.waypoints = np.array([
            [ h,  h], [ h, -h], [-h, -h], [-h,  h],
        ], dtype=np.float64)
        self.current_wp = 0
        self.pos = self.waypoints[0].copy()

        # Obstacles at corners offset inward
        self.obstacles_ned = np.array([
            [ obstacle_range,  obstacle_range],
            [ obstacle_range, -obstacle_range],
            [-obstacle_range, -obstacle_range],
            [-obstacle_range,  obstacle_range],
        ], dtype=np.float32)

    def step(self) -> dict:
        """Advance along square path."""
        self.t += self.dt

        # Move toward current waypoint
        target = self.waypoints[self.current_wp]
        diff = target - self.pos
        dist_to_wp = float(np.linalg.norm(diff))

        if dist_to_wp < 0.1:
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            target = self.waypoints[self.current_wp]
            diff = target - self.pos

        direction = diff / (np.linalg.norm(diff) + 1e-6)
        self.pos += direction * self.speed * self.dt

        pos_ned = np.array([self.pos[0], self.pos[1], -10.0], dtype=np.float32)
        vel_ned = np.array([direction[0] * self.speed,
                            direction[1] * self.speed, 0.0], dtype=np.float32)

        accel = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        accel += self.rng.randn(3).astype(np.float32) * 0.05
        gyro = np.zeros(3, dtype=np.float32)
        gyro += self.rng.randn(3).astype(np.float32) * 0.01

        # Ultrasonic ranges to obstacles
        ranges = []
        bearings = []
        for obs in self.obstacles_ned:
            diff_obs = obs - self.pos.astype(np.float32)
            dist = float(np.linalg.norm(diff_obs))
            bearing = float(np.arctan2(diff_obs[1], diff_obs[0]))
            ranges.append(dist)
            bearings.append(bearing)

        ranges = np.array(ranges, dtype=np.float32)
        bearings = np.array(bearings, dtype=np.float32)

        return {
            'pos_ned': pos_ned,
            'pos_uncertainty': 0.5,
            'accel': accel,
            'gyro': gyro,
            'ranges': ranges,
            'bearings': bearings,
            'min_range': float(np.min(ranges)),
            'vel_ned': vel_ned,
        }

    def reset(self):
        self.t = 0.0
        self.current_wp = 0
        self.pos = self.waypoints[0].copy()


class RewardCalculator:
    """
    Consistent reward signal from sensor state.
    Used identically across all tests for fair comparison.

    reward = +1.0 if min_range > SAFE_RANGE
    reward = -1.0 if min_range <= SAFE_RANGE
    """

    SAFE_RANGE = 1.5  # metres

    @staticmethod
    def compute(min_range: float) -> float:
        return 1.0 if min_range > RewardCalculator.SAFE_RANGE else -1.0


class TrainingRunner:
    """Run N training steps and return metrics."""

    @staticmethod
    def run(controller: FlyBrainController,
            environment,
            n_steps: int,
            sensors_active: dict = None,
            collect_every: int = 100) -> dict:
        """
        Run training loop.

        sensors_active: dict of booleans controlling which sensors inject
            keys: 'gps', 'imu', 'ultrasonic'
            default: all True
        collect_every: record metrics every N steps

        Returns dict with:
            steps:             list of step numbers
            separations:       list of motor separations at each collection point
            mean_weights:      list of mean connection weights
            spike_rates:       list of spike rates
            pos_motor_history: mean motor output when reward > 0, per window
            neg_motor_history: mean motor output when reward < 0, per window
            final_separation:  last separation value
        """
        if sensors_active is None:
            sensors_active = {'gps': True, 'imu': True, 'ultrasonic': True}

        reward_calc = RewardCalculator()

        steps = []
        separations = []
        mean_weights = []
        spike_rates = []
        pos_motor_history = []
        neg_motor_history = []

        window_pos = []
        window_neg = []

        for i in range(n_steps):
            obs = environment.step()
            reward = reward_calc.compute(obs['min_range'])

            # Build kwargs based on active sensors
            kwargs = {'reward': reward, 'dt': 0.01}
            if sensors_active.get('gps', True):
                kwargs['pos_ned'] = obs['pos_ned']
                kwargs['pos_uncertainty'] = obs['pos_uncertainty']
            if sensors_active.get('imu', True):
                kwargs['accel'] = obs['accel']
                kwargs['gyro'] = obs['gyro']
            if sensors_active.get('ultrasonic', True):
                kwargs['ranges'] = obs['ranges']
                kwargs['bearings'] = obs['bearings']

            motors = controller.step(**kwargs)
            mean_motor = float(np.mean(motors))

            if reward > 0:
                window_pos.append(mean_motor)
            else:
                window_neg.append(mean_motor)

            if (i + 1) % collect_every == 0:
                steps.append(i + 1)

                if window_pos and window_neg:
                    sep = abs(np.mean(window_pos) - np.mean(window_neg))
                else:
                    sep = 0.0
                separations.append(float(sep))

                if window_pos:
                    pos_motor_history.append(float(np.mean(window_pos)))
                if window_neg:
                    neg_motor_history.append(float(np.mean(window_neg)))

                mean_weights.append(float(np.mean(controller.plasticity.weights)))
                spike_rates.append(float(controller.torus.spike_rate()))

                window_pos = []
                window_neg = []

        return {
            'steps': steps,
            'separations': separations,
            'mean_weights': mean_weights,
            'spike_rates': spike_rates,
            'pos_motor_history': pos_motor_history,
            'neg_motor_history': neg_motor_history,
            'final_separation': max(separations) if separations else 0.0,
        }
