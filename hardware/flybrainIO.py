# fly_brain/hardware/flybrainIO.py
"""
FlyBrainIO — 100 Hz hardware I/O main loop.

Wires together:
  • UltrasonicArray         — range/bearing measurements
  • FlyBrainController      — wave-navigation brain
  • ESCController           — motor output
  • NavigationResult        — optional dronewarp integration

Timing strategy
---------------
Uses ``threading.Event.wait(timeout)`` with deadline drift correction to
hold 100 Hz regardless of compute jitter.  If a step takes longer than
one tick (10 ms), the scheduler skips the overrun and logs a warning
rather than spiralling into a backlog.

dronewarp integration
---------------------
The loop imports ``dronewarp.NavigationResult`` if available and calls
``controller.step()`` with the reward field populated from the navigation
result's ``reward`` attribute.  If dronewarp is not installed the import
is silently skipped and reward is computed from the raw range readings
(obstacle-clearance heuristic identical to the unit tests).

CSV logging
-----------
Motor commands and reward are appended to a CSV file each tick.
Columns:  t_s, m0, m1, m2, m3, reward, spike_rate

YAML configuration
------------------
Optional config file (default: ``fly_brain_hw.yaml``) follows the same
key structure used by dronewarp configs::

    loop_hz: 100
    sensor:
      n_sensors: 8
      fov_deg: 180
      simulate: true
    esc:
      backend: gpio          # 'gpio' | 'pca9685' | 'simulate'
      motor_pins: [12, 13, 18, 19]
    log:
      csv_path: fly_brain_log.csv
      enabled: true

All keys are optional; defaults match the values in ``fly_brain/config.py``.

Usage
-----
    # Simulation (no hardware required):
    io = FlyBrainIO()
    io.run()                    # runs until Ctrl+C

    # From the command line:
    python -m fly_brain.hardware.flybrainIO
    python -m fly_brain.hardware.flybrainIO --config my_config.yaml --duration 30
"""

import argparse
import csv
import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

from fly_brain.controller import FlyBrainController
from fly_brain.config import (
    N_MOTORS,
    N_ULTRASONIC_SENSORS,
    SAFE_RANGE_M,
    DT,
)
from fly_brain.hardware.ultrasonic_driver import UltrasonicArray
from fly_brain.hardware.esc_output import ESCController

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml          # type: ignore[import-untyped]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from dronewarp import NavigationResult as _NavResult   # type: ignore[import-untyped]
    _DRONEWARP_AVAILABLE = True
except Exception:
    _NavResult = None
    _DRONEWARP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LOOP_HZ:     int   = 100
DEFAULT_CONFIG_PATH: str   = "fly_brain_hw.yaml"
DEFAULT_LOG_PATH:    str   = "fly_brain_log.csv"


# =============================================================================
# Config loader
# =============================================================================

def _load_yaml_config(path: str) -> dict:
    """Load YAML file if available; return empty dict if missing or no yaml lib."""
    if not _YAML_AVAILABLE:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = _yaml.safe_load(fh) or {}
    return data


def _merge(defaults: dict, overrides: dict) -> dict:
    """Shallow-merge two dicts (overrides win)."""
    result = dict(defaults)
    result.update(overrides)
    return result


# =============================================================================
# Reward heuristic (standalone — no dronewarp required)
# =============================================================================

def _clearance_reward(ranges_m: np.ndarray) -> float:
    """
    Simple obstacle-clearance reward in [-1, +1].

    +1 when all readings ≥ SAFE_RANGE_M (open space).
    -1 when the minimum reading is 0 (collision imminent).
    """
    min_range = float(ranges_m.min())
    return float(np.clip(2.0 * min_range / SAFE_RANGE_M - 1.0, -1.0, 1.0))


# =============================================================================
# FlyBrainIO
# =============================================================================

class FlyBrainIO:
    """
    100 Hz flight control loop.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file.  Values in the file override defaults.
    loop_hz : int
        Control loop frequency.  Override YAML ``loop_hz`` key.
    simulate : bool
        Force simulation mode for both sensor and ESC.
    duration_s : float
        Stop after this many seconds (0 = run forever).
    verbose : bool
        Print per-tick status to stdout (default False for 100 Hz loops).
    """

    def __init__(
        self,
        config_path: str  = DEFAULT_CONFIG_PATH,
        loop_hz:     int  = DEFAULT_LOOP_HZ,
        simulate:    bool = True,
        duration_s:  float = 0.0,
        verbose:     bool  = False,
    ) -> None:
        cfg = _load_yaml_config(config_path)

        # Resolved config (YAML overrides constructor defaults)
        self._hz         = int(cfg.get("loop_hz", loop_hz))
        self._tick_s     = 1.0 / self._hz
        self._duration   = float(cfg.get("duration_s", duration_s))
        self._verbose    = verbose

        sensor_cfg = cfg.get("sensor", {})
        esc_cfg    = cfg.get("esc",    {})
        log_cfg    = cfg.get("log",    {})

        sim_sensor = bool(sensor_cfg.get("simulate", simulate))
        sim_esc    = esc_cfg.get("backend", "simulate") == "simulate" or simulate

        # ---- Subsystems ----
        self._sensor = UltrasonicArray(
            n_sensors = int(sensor_cfg.get("n_sensors", N_ULTRASONIC_SENSORS)),
            fov_rad   = math.radians(float(sensor_cfg.get("fov_deg", 180.0))),
            simulate  = sim_sensor,
        )

        self._esc = ESCController(
            motor_pins = tuple(esc_cfg.get("motor_pins", [12, 13, 18, 19])),
            n_motors   = N_MOTORS,
            backend    = esc_cfg.get("backend", "gpio"),
            simulate   = sim_esc,
            verbose    = verbose,
        )

        self._brain = FlyBrainController()

        # ---- CSV logging ----
        self._log_enabled = bool(log_cfg.get("enabled", True))
        self._log_path    = str(log_cfg.get("csv_path", DEFAULT_LOG_PATH))
        self._log_file    = None
        self._log_writer  = None

        # ---- Threading ----
        self._stop_event  = threading.Event()
        self._step_count  = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Arm the ESC then enter the 100 Hz control loop.

        Blocks until ``duration_s`` has elapsed or Ctrl+C is pressed.
        Calls ``shutdown()`` automatically on exit.
        """
        print(f"[FlyBrainIO] Starting at {self._hz} Hz", flush=True)
        if _DRONEWARP_AVAILABLE:
            print("[FlyBrainIO] dronewarp integration: active", flush=True)
        else:
            print("[FlyBrainIO] dronewarp not found — using clearance reward heuristic",
                  flush=True)

        self._open_log()
        self._esc.arm()

        t_start  = time.monotonic()
        deadline = t_start + self._tick_s

        try:
            while not self._stop_event.is_set():
                # ---- duration check ----
                now = time.monotonic()
                elapsed = now - t_start
                if self._duration > 0 and elapsed >= self._duration:
                    break

                # ---- control tick ----
                self._tick(elapsed)

                # ---- deadline scheduling ----
                now = time.monotonic()
                sleep_s = deadline - now
                if sleep_s > 0.0:
                    self._stop_event.wait(sleep_s)
                else:
                    # Overrun: skip to next whole-tick boundary to avoid drift
                    missed = int(-sleep_s / self._tick_s) + 1
                    if missed > 1:
                        print(f"[FlyBrainIO] WARNING: overrun by {-sleep_s*1000:.1f} ms "
                              f"({missed} ticks skipped)", flush=True)
                    deadline += missed * self._tick_s
                    continue

                deadline += self._tick_s

        except KeyboardInterrupt:
            print("\n[FlyBrainIO] Stopped by user.", flush=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Disarm ESC, release GPIO, and close log file."""
        self._stop_event.set()
        self._esc.close()
        self._sensor.close()
        self._close_log()
        print(f"[FlyBrainIO] Shutdown complete.  "
              f"Total steps: {self._step_count}", flush=True)

    def request_stop(self) -> None:
        """Thread-safe stop request (e.g. from a signal handler)."""
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Single tick                                                          #
    # ------------------------------------------------------------------ #

    def _tick(self, elapsed_s: float) -> None:
        """Execute one control cycle."""
        # 1. Sensor measurement
        ranges_m, bearings_rad = self._sensor.measure()

        # 2. Reward signal
        if _DRONEWARP_AVAILABLE:
            nav_result: "_NavResult" = _NavResult(ranges=ranges_m,
                                                   bearings=bearings_rad)
            reward = float(getattr(nav_result, "reward", _clearance_reward(ranges_m)))
        else:
            reward = _clearance_reward(ranges_m)

        # 3. Brain step — gyro/accel left as None (only ultrasonic in this loop)
        motor_cmds = self._brain.step(
            ranges=ranges_m,
            bearings=bearings_rad,
            reward=reward,
            dt=self._tick_s,
        )

        # 4. ESC output — motor_cmds are already in [0,1] from MotorReadout
        self._esc.send(motor_cmds)

        self._step_count += 1

        # 5. Logging
        if self._log_enabled and self._log_writer is not None:
            row = [f"{elapsed_s:.6f}"] + [f"{c:.6f}" for c in motor_cmds]
            row += [f"{reward:.6f}"]
            # Append spike rate from ocellus torus for diagnostics
            spike_rate = float(
                np.mean(self._brain.torus_ocellus.spikes) if hasattr(
                    self._brain.torus_ocellus, "spikes") else 0.0
            )
            row.append(f"{spike_rate:.6f}")
            self._log_writer.writerow(row)

        # 6. Verbose status
        if self._verbose:
            print(f"\r[FlyBrainIO] t={elapsed_s:7.3f}s  "
                  f"range_min={ranges_m.min():.2f}m  "
                  f"reward={reward:+.3f}  "
                  f"motors={np.round(motor_cmds, 2).tolist()}",
                  end="", flush=True)

    # ------------------------------------------------------------------ #
    # CSV helpers                                                          #
    # ------------------------------------------------------------------ #

    def _open_log(self) -> None:
        if not self._log_enabled:
            return
        self._log_file   = open(self._log_path, "w", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        header = ["t_s"] + [f"m{i}" for i in range(N_MOTORS)] + ["reward", "spike_rate"]
        self._log_writer.writerow(header)

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file   = None
            self._log_writer = None
            print(f"[FlyBrainIO] Log saved → {self._log_path}", flush=True)


# =============================================================================
# CLI entry point
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlyBrain hardware I/O loop — run at 100 Hz."
    )
    p.add_argument("--config",    default=DEFAULT_CONFIG_PATH,
                   help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--hz",        type=int,   default=DEFAULT_LOOP_HZ,
                   help="Loop frequency in Hz (default: 100)")
    p.add_argument("--duration",  type=float, default=0.0,
                   help="Run for this many seconds then stop (0 = forever)")
    p.add_argument("--simulate",  action="store_true",
                   help="Force simulation mode for sensor + ESC")
    p.add_argument("--verbose",   action="store_true",
                   help="Print status each tick")
    p.add_argument("--no-log",    action="store_true",
                   help="Disable CSV logging")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    io = FlyBrainIO(
        config_path = args.config,
        loop_hz     = args.hz,
        simulate    = args.simulate,
        duration_s  = args.duration,
        verbose     = args.verbose,
    )
    if args.no_log:
        io._log_enabled = False

    io.run()
