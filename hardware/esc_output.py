# fly_brain/hardware/esc_output.py
"""
ESCController — Electronic Speed Controller output driver.

Converts normalised throttle commands in [0.0, 1.0] to ESC PWM pulses in
the standard 1000–2000 µs range used by brushless motor controllers.

Two backend modes
-----------------
GPIO PWM (default)
    Uses the RPi hardware PWM or software PWM on a configurable GPIO pin
    per motor.  Suitable for direct connections to up to 4 ESCs.

PCA9685 (I2C PWM driver)
    16-channel I2C PWM board at a configurable I2C address (default 0x40).
    Useful when the Pi's native PWM pins are occupied by other peripherals.
    Requires:  pip install adafruit-circuitpython-pca9685

Simulation mode
---------------
Neither backend is instantiated when RPi.GPIO and adafruit libs are absent.
In simulation mode all commands are printed to stdout (optionally silenced)
and the internal ``_throttles`` array reflects the last sent command.

Safety
------
* All commands are hard-clipped to [0.0, 1.0] before conversion.
* An arming sequence sends minimum throttle (1000 µs) for ARM_DURATION_S
  seconds on the first call to ``arm()``.  The controller will not pass any
  command until arming completes.
* ``disarm()`` sends minimum throttle to all channels and marks unarmed.

Public API
----------
    esc = ESCController(motor_pins=[12, 13, 18, 19])  # GPIO BCM pins
    esc = ESCController(backend='pca9685')
    esc = ESCController(simulate=True)

    esc.arm()                          # blocking — waits ARM_DURATION_S
    esc.send(np.array([0.5, 0.5, 0.5, 0.5]))  # normalised [0,1]
    esc.disarm()
    esc.close()
"""

import time
import numpy as np

from fly_brain.config import (
    N_MOTORS,
    ESC_MIN_PULSE_US,
    ESC_MAX_PULSE_US,
    PWM_FREQUENCY_HZ,
)

# ---------------------------------------------------------------------------
# Backend availability probes — never raise on non-Pi hardware
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as _GPIO              # type: ignore[import-untyped]
    _GPIO.setmode(_GPIO.BCM)
    _GPIO_AVAILABLE = True
except Exception:
    _GPIO_AVAILABLE = False
    _GPIO = None

try:
    import board as _board                # type: ignore[import-untyped]
    import busio as _busio                # type: ignore[import-untyped]
    from adafruit_pca9685 import PCA9685 as _PCA9685   # type: ignore[import-untyped]
    _PCA9685_AVAILABLE = True
except Exception:
    _PCA9685_AVAILABLE = False
    _PCA9685 = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARM_DURATION_S:   float = 2.0    # seconds of minimum-throttle arming
_PERIOD_US:       float = 1e6 / PWM_FREQUENCY_HZ   # µs per PWM cycle
_PCA9685_MAX_CNT: int   = 4095   # 12-bit resolution

# Default GPIO pins (BCM) for 4 motors — matches standard Pi quadrotor wiring
_DEFAULT_MOTOR_PINS = (12, 13, 18, 19)


# ---------------------------------------------------------------------------
# Helper: convert [0,1] normalised throttle → pulse width in µs
# ---------------------------------------------------------------------------

def _throttle_to_us(throttle: float) -> float:
    """Map [0.0, 1.0] → [ESC_MIN_PULSE_US, ESC_MAX_PULSE_US]."""
    t = float(np.clip(throttle, 0.0, 1.0))
    return ESC_MIN_PULSE_US + t * (ESC_MAX_PULSE_US - ESC_MIN_PULSE_US)


def _us_to_duty_cycle(pulse_us: float) -> float:
    """Convert pulse width in µs → duty cycle fraction [0, 1]."""
    return pulse_us / _PERIOD_US


def _us_to_pca9685_count(pulse_us: float) -> int:
    """Convert pulse width in µs → PCA9685 on-time count (0–4095)."""
    return int(round(pulse_us / _PERIOD_US * _PCA9685_MAX_CNT))


# =============================================================================
# ESCController
# =============================================================================

class ESCController:
    """
    Multi-channel ESC driver for a quadrotor.

    Parameters
    ----------
    motor_pins : sequence of int, optional
        BCM GPIO pin numbers for software/hardware PWM output, one per motor.
        Used only when ``backend='gpio'``.
    n_motors : int
        Number of motors.  Must match ``len(motor_pins)`` in GPIO mode.
        Defaults to ``N_MOTORS`` from config (4).
    backend : {'gpio', 'pca9685'}
        Select output backend.  Silently falls back to simulation if the
        chosen backend's hardware is not available.
    i2c_address : int
        I2C address of the PCA9685 board (default 0x40).
    simulate : bool
        Force simulation mode regardless of hardware availability.
    verbose : bool
        Print throttle commands to stdout in simulation mode (default True).
    """

    def __init__(
        self,
        motor_pins:  tuple    = _DEFAULT_MOTOR_PINS,
        n_motors:    int      = N_MOTORS,
        backend:     str      = "gpio",
        i2c_address: int      = 0x40,
        simulate:    bool     = False,
        verbose:     bool     = True,
    ) -> None:
        if backend not in ("gpio", "pca9685"):
            raise ValueError(f"backend must be 'gpio' or 'pca9685', got {backend!r}")

        self._n       = n_motors
        self._verbose = verbose
        self._armed   = False
        self._throttles = np.zeros(n_motors, dtype=np.float32)

        # Decide actual operating mode
        if simulate:
            self._mode = "simulate"
        elif backend == "pca9685" and _PCA9685_AVAILABLE:
            self._mode = "pca9685"
        elif backend == "gpio" and _GPIO_AVAILABLE:
            self._mode = "gpio"
        else:
            self._mode = "simulate"
            if not simulate:
                print(f"[ESCController] {backend} backend not available "
                      f"— running in simulation mode.")

        # Backend initialisation
        if self._mode == "gpio":
            self._pins = list(motor_pins[:n_motors])
            self._pwm_channels = []
            for pin in self._pins:
                _GPIO.setup(pin, _GPIO.OUT)
                pwm = _GPIO.PWM(pin, PWM_FREQUENCY_HZ)
                pwm.start(0.0)          # start at 0% duty cycle
                self._pwm_channels.append(pwm)

        elif self._mode == "pca9685":
            i2c = _busio.I2C(_board.SCL, _board.SDA)
            self._pca = _PCA9685(i2c, address=i2c_address)
            self._pca.frequency = PWM_FREQUENCY_HZ
            # Motors use channels 0..n_motors-1
            self._pca_channels = [self._pca.channels[i] for i in range(n_motors)]

    # ------------------------------------------------------------------ #
    # Arming / disarming                                                   #
    # ------------------------------------------------------------------ #

    def arm(self) -> None:
        """
        Send minimum throttle for ARM_DURATION_S to allow ESC initialisation.

        This method blocks for ARM_DURATION_S seconds.  Call it once at
        startup before sending any throttle commands.
        """
        if self._armed:
            return
        print(f"[ESCController] Arming — sending min throttle for {ARM_DURATION_S:.1f} s …",
              flush=True)
        min_cmd = np.zeros(self._n, dtype=np.float32)
        self._write(min_cmd)
        time.sleep(ARM_DURATION_S)
        self._armed = True
        print("[ESCController] Armed.", flush=True)

    def disarm(self) -> None:
        """Send minimum throttle to all channels and mark as unarmed."""
        self._write(np.zeros(self._n, dtype=np.float32))
        self._armed = False
        print("[ESCController] Disarmed.", flush=True)

    # ------------------------------------------------------------------ #
    # Command sending                                                      #
    # ------------------------------------------------------------------ #

    def send(self, throttles: np.ndarray) -> np.ndarray:
        """
        Set motor throttles.

        Parameters
        ----------
        throttles : array-like of float, shape (n_motors,)
            Normalised throttle commands in [0.0, 1.0].  Values outside this
            range are silently clipped.

        Returns
        -------
        throttles_clipped : (n_motors,) float32 — the actual commands sent.

        Raises
        ------
        RuntimeError
            If the controller has not been armed yet.
        """
        if not self._armed:
            raise RuntimeError(
                "ESCController.send() called before arm(). "
                "Call esc.arm() once at startup."
            )
        cmds = np.clip(np.asarray(throttles, dtype=np.float32), 0.0, 1.0)
        if len(cmds) != self._n:
            raise ValueError(
                f"Expected {self._n} throttle commands, got {len(cmds)}."
            )
        self._write(cmds)
        return cmds

    # ------------------------------------------------------------------ #
    # Internal write                                                       #
    # ------------------------------------------------------------------ #

    def _write(self, throttles: np.ndarray) -> None:
        """Write throttle array to the active backend (no safety gating)."""
        self._throttles[:] = throttles

        if self._mode == "gpio":
            for i, (pwm, t) in enumerate(zip(self._pwm_channels, throttles)):
                duty = _us_to_duty_cycle(_throttle_to_us(float(t))) * 100.0
                pwm.ChangeDutyCycle(duty)

        elif self._mode == "pca9685":
            for ch, t in zip(self._pca_channels, throttles):
                cnt = _us_to_pca9685_count(_throttle_to_us(float(t)))
                ch.duty_cycle = int(cnt * 65535 // _PCA9685_MAX_CNT)

        elif self._mode == "simulate" and self._verbose:
            pulse_us = [int(_throttle_to_us(float(t))) for t in throttles]
            print(f"[ESC sim] throttles={np.round(throttles, 3).tolist()}  "
                  f"pulses_us={pulse_us}")

    # ------------------------------------------------------------------ #
    # State query                                                          #
    # ------------------------------------------------------------------ #

    @property
    def armed(self) -> bool:
        """True if the arm sequence has completed."""
        return self._armed

    @property
    def throttles(self) -> np.ndarray:
        """Last commanded throttle values, shape (n_motors,)."""
        return self._throttles.copy()

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Disarm and release hardware resources."""
        if self._armed:
            self.disarm()

        if self._mode == "gpio":
            for pwm in self._pwm_channels:
                pwm.stop()
            for pin in self._pins:
                _GPIO.cleanup(pin)

        elif self._mode == "pca9685":
            # Zero all channels before releasing I2C bus
            for ch in self._pca_channels:
                ch.duty_cycle = 0
            self._pca.deinit()
