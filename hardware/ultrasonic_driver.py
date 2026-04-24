# fly_brain/hardware/ultrasonic_driver.py
"""
UltrasonicArray — HC-SR04 sensor array driver.

Supports up to N_ULTRASONIC_SENSORS (8) HC-SR04 sensors wired to
Raspberry Pi GPIO pins.  Each sensor measures distance via echo timing.

Hardware wiring (per sensor)
-----------------------------
  TRIG pin  — output, driven HIGH for 10 µs to start a measurement
  ECHO pin  — input, measures the round-trip travel time of the 40 kHz burst

Distance formula
----------------
  d = t_echo * SPEED_OF_SOUND_M_S / 2

Bearings
--------
Sensors are arranged in a horizontal arc.  The bearing for sensor i is:

  bearing_i = -FOV/2 + i * FOV/(n_sensors - 1)    [radians]

where FOV is the total field of view of the array (default π rad = 180°).

Simulation mode
---------------
If RPi.GPIO is not importable (running on a non-Pi machine) the class
falls back to CircularEnvironment, which places random cylindrical
obstacles and returns synthetic range readings that respond realistically
to the drone's simulated position.  All public API is identical.

Public API
----------
  array = UltrasonicArray(trigger_pins, echo_pins)   # hardware
  array = UltrasonicArray(simulate=True)             # simulation
  ranges_m, bearings_rad = array.measure()           # (n,) each
  array.close()
"""

import math
import time
import threading
import numpy as np

from fly_brain.config import (
    N_ULTRASONIC_SENSORS,
    SAFE_RANGE_M,
)

# ---------------------------------------------------------------------------
# GPIO availability check — never fails on non-Pi hardware
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as _GPIO          # type: ignore[import-untyped]
    _GPIO.setmode(_GPIO.BCM)
    _GPIO_AVAILABLE = True
except Exception:                     # ImportError, RuntimeError on non-Pi
    _GPIO_AVAILABLE = False
    _GPIO = None                      # sentinel

# Physical constants
_SPEED_OF_SOUND_M_S: float = 343.0   # m/s at 20 °C

# HC-SR04 timing
_TRIGGER_PULSE_S:    float = 10e-6   # 10 µs trigger pulse
_ECHO_TIMEOUT_S:     float = 0.030   # 30 ms ≈ 5.15 m max range
_MAX_RANGE_M:        float = 4.0     # clip readings beyond this

# Default pins (BCM numbering on a Pi Zero 2W)
_DEFAULT_TRIGGER_PINS = (17, 27, 22, 10, 9, 11, 5, 6)   # TRIG
_DEFAULT_ECHO_PINS    = (18, 23, 24, 25, 8, 7, 12, 16)  # ECHO


# =============================================================================
# Circular obstacle environment used in simulation mode
# =============================================================================

class CircularEnvironment:
    """
    A 2-D planar environment with cylindrical obstacles for unit testing.

    Obstacles are placed randomly on construction and persist for the life
    of the environment.  ``query_ranges`` returns the slant range to the
    nearest obstacle in each bearing direction from a given position.

    Parameters
    ----------
    n_obstacles : int
        Number of cylindrical obstacles to place.
    world_radius : float
        Half-width of the square arena in metres.
    obstacle_radius : float
        Radius of each cylindrical obstacle in metres.
    seed : int, optional
        Random seed for reproducible layouts.
    """

    def __init__(
        self,
        n_obstacles: int = 12,
        world_radius: float = 10.0,
        obstacle_radius: float = 0.3,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Place obstacles avoiding the centre (drone spawn zone)
        candidates = rng.uniform(-world_radius, world_radius, (n_obstacles * 4, 2))
        far_enough = np.linalg.norm(candidates, axis=1) > 2.0
        self._centres = candidates[far_enough][:n_obstacles]
        self._r       = obstacle_radius
        self._world_r = world_radius

    def query_ranges(
        self,
        pos_xy: np.ndarray,
        bearings_rad: np.ndarray,
        max_range: float = _MAX_RANGE_M,
    ) -> np.ndarray:
        """
        Ray-cast from pos_xy along each bearing; return range to nearest
        obstacle (or max_range if no hit within range).

        Parameters
        ----------
        pos_xy      : (2,) array — drone XY position in metres
        bearings_rad: (n,) array — bearings in radians (0 = forward/+X axis)
        max_range   : float      — cap return value at this distance

        Returns
        -------
        ranges : (n,) float32 array
        """
        pos = np.asarray(pos_xy, dtype=np.float64)
        ranges = np.full(len(bearings_rad), max_range, dtype=np.float32)

        for i, b in enumerate(bearings_rad):
            dx = math.cos(b)
            dy = math.sin(b)
            for cx, cy in self._centres:
                # Ray–circle intersection (analytic, no sqrt unless needed)
                fx = pos[0] - cx
                fy = pos[1] - cy
                # a = dx²+dy² = 1  (unit ray)
                half_b = fx * dx + fy * dy
                c = fx * fx + fy * fy - self._r * self._r
                discriminant = half_b * half_b - c
                if discriminant < 0.0:
                    continue           # ray misses cylinder
                sqrt_d = math.sqrt(discriminant)
                t1 = -half_b - sqrt_d
                t2 = -half_b + sqrt_d
                t = t1 if t1 > 1e-4 else (t2 if t2 > 1e-4 else None)
                if t is not None and t < ranges[i]:
                    ranges[i] = float(t)

        return ranges


# =============================================================================
# UltrasonicArray
# =============================================================================

class UltrasonicArray:
    """
    Array of HC-SR04 ultrasonic range sensors.

    Parameters
    ----------
    trigger_pins : sequence of int, optional
        BCM GPIO pin numbers for TRIG signals.  Length determines actual
        sensor count.  Defaults to _DEFAULT_TRIGGER_PINS[:n_sensors].
    echo_pins : sequence of int, optional
        BCM GPIO pin numbers for ECHO signals.  Must match trigger_pins length.
    n_sensors : int
        Number of sensors to use.  Clipped to len(trigger_pins) and
        N_ULTRASONIC_SENSORS.
    fov_rad : float
        Total angular field of view of the array in radians.  Sensors are
        spread evenly across this arc.  Default π (180°).
    simulate : bool
        If True, force simulation mode regardless of GPIO availability.
    env : CircularEnvironment, optional
        Provide a custom environment for simulation mode.  If None a default
        environment is created automatically.
    """

    def __init__(
        self,
        trigger_pins: tuple = _DEFAULT_TRIGGER_PINS,
        echo_pins:    tuple = _DEFAULT_ECHO_PINS,
        n_sensors:    int   = N_ULTRASONIC_SENSORS,
        fov_rad:      float = math.pi,
        simulate:     bool  = False,
        env: "CircularEnvironment | None" = None,
    ) -> None:
        self._n       = min(n_sensors, N_ULTRASONIC_SENSORS, len(trigger_pins), len(echo_pins))
        self._fov     = fov_rad
        self._simulate = simulate or not _GPIO_AVAILABLE

        # Bearing for each sensor (evenly spaced across FOV)
        if self._n > 1:
            self._bearings = np.linspace(-fov_rad / 2.0, fov_rad / 2.0,
                                         self._n, dtype=np.float32)
        else:
            self._bearings = np.zeros(1, dtype=np.float32)

        if self._simulate:
            self._env = env if env is not None else CircularEnvironment()
            # Simulated drone position (updated externally for testing)
            self._sim_pos = np.zeros(2, dtype=np.float64)
            if not _GPIO_AVAILABLE and not simulate:
                print("[UltrasonicArray] RPi.GPIO not available — running in simulation mode.")
        else:
            # Hardware setup
            self._trig = list(trigger_pins[: self._n])
            self._echo = list(echo_pins[: self._n])
            for pin in self._trig:
                _GPIO.setup(pin, _GPIO.OUT, initial=_GPIO.LOW)
            for pin in self._echo:
                _GPIO.setup(pin, _GPIO.IN)
            # One lock per sensor so measurements can parallelise
            self._locks = [threading.Lock() for _ in range(self._n)]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_sensors(self) -> int:
        """Actual number of active sensors."""
        return self._n

    @property
    def bearings(self) -> np.ndarray:
        """Bearing of each sensor in radians.  Shape (n_sensors,)."""
        return self._bearings.copy()

    def measure(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Trigger all sensors and return (ranges_m, bearings_rad).

        In hardware mode sensors are fired sequentially to avoid acoustic
        cross-talk (parallel firing would cause false echoes).

        Returns
        -------
        ranges_m    : (n_sensors,) float32 — range in metres, clipped to [0, MAX_RANGE]
        bearings_rad: (n_sensors,) float32 — bearing of each reading in radians
        """
        if self._simulate:
            return self._measure_simulated()
        return self._measure_hardware()

    def set_sim_position(self, x: float, y: float) -> None:
        """
        Update the drone's 2-D position used by simulation mode.

        Has no effect in hardware mode.
        """
        self._sim_pos[0] = x
        self._sim_pos[1] = y

    def close(self) -> None:
        """Release GPIO resources.  No-op in simulation mode."""
        if not self._simulate and _GPIO_AVAILABLE:
            for pin in self._trig + self._echo:
                _GPIO.cleanup(pin)

    # ------------------------------------------------------------------ #
    # Hardware measurement                                                 #
    # ------------------------------------------------------------------ #

    def _measure_one(self, idx: int) -> float:
        """Fire sensor *idx* and return the measured range in metres."""
        trig = self._trig[idx]
        echo = self._echo[idx]

        with self._locks[idx]:
            # Send 10 µs trigger pulse
            _GPIO.output(trig, _GPIO.HIGH)
            time.sleep(_TRIGGER_PULSE_S)
            _GPIO.output(trig, _GPIO.LOW)

            # Wait for echo to go HIGH (start of return pulse)
            t_deadline = time.monotonic() + _ECHO_TIMEOUT_S
            while _GPIO.input(echo) == _GPIO.LOW:
                if time.monotonic() > t_deadline:
                    return _MAX_RANGE_M   # timeout

            t_start = time.monotonic()

            # Wait for echo to go LOW (end of return pulse)
            t_deadline = t_start + _ECHO_TIMEOUT_S
            while _GPIO.input(echo) == _GPIO.HIGH:
                if time.monotonic() > t_deadline:
                    return _MAX_RANGE_M

            t_echo = time.monotonic() - t_start

        distance = t_echo * _SPEED_OF_SOUND_M_S / 2.0
        return float(np.clip(distance, 0.0, _MAX_RANGE_M))

    def _measure_hardware(self) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.array(
            [self._measure_one(i) for i in range(self._n)],
            dtype=np.float32,
        )
        return ranges, self._bearings.copy()

    # ------------------------------------------------------------------ #
    # Simulation measurement                                               #
    # ------------------------------------------------------------------ #

    def _measure_simulated(self) -> tuple[np.ndarray, np.ndarray]:
        """Ray-cast into CircularEnvironment from current sim position."""
        ranges = self._env.query_ranges(self._sim_pos, self._bearings)
        # Add small Gaussian noise to mimic real sensor jitter (±1 cm)
        noise  = np.random.randn(self._n).astype(np.float32) * 0.01
        ranges = np.clip(ranges + noise, 0.0, _MAX_RANGE_M).astype(np.float32)
        return ranges, self._bearings.copy()
