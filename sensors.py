# fly_brain/sensors.py
"""
Sensor-to-wave injection layer.

Maps physical sensor readings (GPS, IMU, ultrasonic) to wave patterns
injected into dedicated regions of the torus.

Biological analogy:
  - GPS → visual place cells (hippocampus-like position encoding)
  - IMU → Johnston's organ in the fly (encodes self-motion as vibration)
  - Ultrasonic → bat auditory cortex (echo timing encodes obstacle distance)
  - Reward → dopamine neuromodulation (global excitatory/inhibitory pulse)
"""

import numpy as np
from .torus import FlyBrainTorus
from .config import (
    TORUS_W, TORUS_H,
    GPS_REGION, IMU_REGION, SONIC_REGION,
    V_CRIT,
)


class SensorInjector:
    """
    Converts sensor readings to wave patterns for torus injection.

    Design principle: each sensor type maps to a dedicated torus region.
    The spatial pattern of injection encodes the sensor value.
    High value = large amplitude injection at region centre.
    Direction = phase gradient across region.
    """

    # GPS position clamp (metres) — values outside this range get saturated
    GPS_POS_RANGE: float = 50.0

    # IMU clamp values
    IMU_ACCEL_MAX: float = 20.0   # m/s² (covers ±2g with margin)
    IMU_GYRO_MAX:  float = 10.0   # rad/s

    # Ultrasonic clamp
    SONIC_MAX_RANGE: float = 10.0  # metres

    # Reward modulation amplitude
    REWARD_AMPLITUDE: float = 0.5

    def inject_gps(
        self,
        torus: FlyBrainTorus,
        pos_ned: np.ndarray,
        uncertainty: float = 5.0,
    ) -> None:
        """
        Inject GPS position as wave into GPS_REGION.

        pos_ned: [N, E, D] in metres (local NED frame)
        uncertainty: 1-sigma position uncertainty in metres

        Encoding:
        - N position → injection x-coordinate within GPS region
        - E position → injection y-coordinate within GPS region
        - D (down) → amplitude modulation
        - Amplitude ∝ 1/uncertainty (confident GPS = strong wave)

        Biological analogy: visual place cells — each cell fires for a
        specific location. Here the injection point moves as position changes.
        """
        pos = np.asarray(pos_ned, dtype=np.float32)
        x0, y0, x1, y1 = GPS_REGION
        rw = x1 - x0
        rh = y1 - y0

        # Map N → x within region, E → y within region
        n_norm = float(np.clip(pos[0], -self.GPS_POS_RANGE, self.GPS_POS_RANGE))
        e_norm = float(np.clip(pos[1], -self.GPS_POS_RANGE, self.GPS_POS_RANGE))

        ix = int((n_norm / self.GPS_POS_RANGE + 1.0) * 0.5 * (rw - 1))
        iy = int((e_norm / self.GPS_POS_RANGE + 1.0) * 0.5 * (rh - 1))
        ix = int(np.clip(ix, 0, rw - 1))
        iy = int(np.clip(iy, 0, rh - 1))

        amplitude = float(np.clip(1.0 / max(uncertainty, 0.1), 0.0, 10.0))

        # Inject a Gaussian blob centred at (ix, iy) in the region
        pattern = self._gaussian_blob(rh, rw, iy, ix, sigma=2.0) * amplitude
        torus.inject_region(GPS_REGION, pattern)

    def inject_halteres(
        self,
        torus: FlyBrainTorus,
        gyro: np.ndarray,
    ) -> None:
        """
        Inject halteres signal (angular velocity only) into the 0-pole torus.

        Biological accuracy: halteres detect ONLY rotation via Coriolis forces.
        They are insensitive to linear acceleration.  This separation from
        inject_johnston (linear accel) mirrors the biological split between
        halteres and Johnston's organ.

        gyro: [gx, gy, gz] in rad/s — angular velocity vector

        Encoding:
          - |gyro| -> injection amplitude (rotation magnitude)
          - gyro direction -> centroid in IMU_REGION
          - Phase ramp across region encodes rotation rate (as in inject_imu)
        """
        gyro = np.asarray(gyro, dtype=np.float32)
        self.inject_imu(torus, accel=np.zeros(3, dtype=np.float32), gyro=gyro)

    def inject_johnston(
        self,
        torus: FlyBrainTorus,
        accel: np.ndarray,
    ) -> None:
        """
        Inject Johnston's organ signal (linear acceleration) into the antenna torus.

        Biological accuracy: Johnston's organ (at the antenna base) detects
        near-field air particle velocity and substrate vibration — which in
        our sensor model corresponds to linear acceleration.  It is the
        vibration/acceleration counterpart to the halteres' rotation sensing.

        accel: [ax, ay, az] in m/s^2 — linear acceleration vector

        Encoding: same as inject_imu but with zero gyro — amplitude encodes
        |accel| magnitude, centroid encodes accel direction within IMU_REGION.
        Adds a small proximity-correlated pattern to the antenna (-1 pole) torus:
        deceleration near walls is a proxy for obstacle proximity.
        """
        accel = np.asarray(accel, dtype=np.float32)
        self.inject_imu(torus, accel=accel, gyro=np.zeros(3, dtype=np.float32))

    def inject_imu(
        self,
        torus: FlyBrainTorus,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        """
        Inject IMU measurements as frequency-modulated wave into IMU_REGION.

        accel: [ax, ay, az] in m/s²
        gyro:  [gx, gy, gz] in rad/s

        Encoding:
        - |accel| → injection amplitude
        - accel direction (normalised) → injection centroid offset within region
        - |gyro|  → phase offset of the injected pattern (frequency modulation)

        Biological analogy: fly Johnston's organ encodes body rotation and
        wind via mechanosensory neurons. Each neuron is tuned to a direction;
        here the injection position encodes direction and amplitude encodes magnitude.
        """
        accel = np.asarray(accel, dtype=np.float32)
        gyro  = np.asarray(gyro,  dtype=np.float32)

        x0, y0, x1, y1 = IMU_REGION
        rw = x1 - x0
        rh = y1 - y0

        accel_mag = float(np.linalg.norm(accel))
        gyro_mag  = float(np.linalg.norm(gyro))

        # Normalised accel direction maps to centroid in region
        if accel_mag > 1e-6:
            a_norm = accel / accel_mag
        else:
            a_norm = np.zeros(3, dtype=np.float32)

        # Use accel x,y components to place centroid; z modulates amplitude
        cx = int(np.clip((a_norm[0] * 0.5 + 0.5) * (rw - 1), 0, rw - 1))
        cy = int(np.clip((a_norm[1] * 0.5 + 0.5) * (rh - 1), 0, rh - 1))

        amplitude = float(np.clip(accel_mag / self.IMU_ACCEL_MAX, 0.0, 1.0))

        # Gyro adds a phase-gradient across the region
        gyro_phase = float(np.clip(gyro_mag / self.IMU_GYRO_MAX, 0.0, 1.0))

        pattern = self._gaussian_blob(rh, rw, cy, cx, sigma=2.0) * amplitude

        # Add a linear phase ramp driven by gyro magnitude
        yy, xx = np.meshgrid(
            np.linspace(0, gyro_phase, rh),
            np.linspace(0, gyro_phase, rw),
            indexing='ij',
        )
        pattern = pattern + 0.1 * (yy + xx).astype(np.float32)

        torus.inject_region(IMU_REGION, pattern)

    def inject_clearance(
        self,
        torus: FlyBrainTorus,
        ranges: np.ndarray,
        bearings: np.ndarray,
    ) -> None:
        """
        Inject clearance map into the +1 (safe) torus phase.

        This is the COMPLEMENT of inject_ultrasonic: the same bearing→position
        mapping, but amplitude ∝ (SAFE_RANGE - r) / SAFE_RANGE.
        High amplitude = obstacle far from SAFE_RANGE = extra-safe zone.
        Zero amplitude = obstacle at SAFE_RANGE = boundary.
        Negative amplitude clips to 0 = obstacle closer than SAFE_RANGE = danger.

        Together with inject_ultrasonic on the −1 (sonic) torus, this forms
        a balanced ternary pair:
            V_in_safe  − V_in_sonic  > 0  →  clear beyond SAFE_RANGE  (reward +1 region)
            V_in_safe  − V_in_sonic  < 0  →  closer than SAFE_RANGE   (reward −1 region)
            Crossover at r = SAFE_RANGE

        Because the two signals use identical spatial encoding and share a common
        crossover point, the ternary difference directly tracks the reward signal —
        the readout can learn the mapping in ~100 steps.

        GPS is NOT injected here. GPS should only be used when a pre-mapped
        obstacle database is available (look up GPS coordinate → inject known
        obstacle bearings/ranges from that database into this torus).
        """
        from .config import SAFE_RANGE
        
        ranges   = np.asarray(ranges,   dtype=np.float32).ravel()
        bearings = np.asarray(bearings, dtype=np.float32).ravel()

        x0, y0, x1, y1 = SONIC_REGION   # same spatial layout as sonic torus
        rw = x1 - x0
        rh = y1 - y0

        pattern = np.zeros((rh, rw), dtype=np.float32)

        for r, b in zip(ranges, bearings):
            r = float(np.clip(r, 0.0, self.SONIC_MAX_RANGE))
            b = float(b)

            # Clearance amplitude: HIGH when far from SAFE_RANGE, ZERO at boundary
            # Aligned crossover: trit = 0 exactly when r = SAFE_RANGE
            amplitude = float(np.clip(
                V_CRIT * (1.0 - r / SAFE_RANGE), 0.0, V_CRIT
            ))

            # Bearing → horizontal, range → vertical (same as sonic for symmetry)
            bx = int(np.clip(
                (b / (2 * np.pi) + 0.5) % 1.0 * (rw - 1), 0, rw - 1
            ))
            by = int(np.clip(
                (r / self.SONIC_MAX_RANGE) * (rh - 1), 0, rh - 1
            ))
            pattern += self._gaussian_blob(rh, rw, by, bx, sigma=1.5) * amplitude

        torus.inject_region(SONIC_REGION, pattern)

    def inject_ultrasonic(
        self,
        torus: FlyBrainTorus,
        ranges: np.ndarray,
        bearings: np.ndarray,
    ) -> None:
        """
        Inject ranging data as echo-like wave pattern into SONIC_REGION.

        ranges:   distances to obstacles in metres (1D array)
        bearings: bearing to each obstacle in radians (1D array, same length)

        Encoding:
        - Short range → high amplitude (obstacle close = strong echo)
        - Bearing → spatial position within sonic region (left/right = angle)
        - Amplitude = V_CRIT * (1 - range/SONIC_MAX_RANGE) → saturates at very close

        Biological analogy: bat auditory cortex encodes echo delay (distance)
        and Doppler shift (bearing/velocity). Close obstacles create strong
        short-delay echoes. The spatial map in SONIC_REGION mirrors this tonotopic
        / azimuthal organisation.
        """
        ranges   = np.asarray(ranges,   dtype=np.float32).ravel()
        bearings = np.asarray(bearings, dtype=np.float32).ravel()

        x0, y0, x1, y1 = SONIC_REGION
        rw = x1 - x0
        rh = y1 - y0

        pattern = np.zeros((rh, rw), dtype=np.float32)

        for r, b in zip(ranges, bearings):
            r = float(np.clip(r, 0.0, self.SONIC_MAX_RANGE))
            b = float(b)

            # Amplitude inversely proportional to range
            amplitude = float(np.clip(
                V_CRIT * (1.0 - r / self.SONIC_MAX_RANGE), 0.0, V_CRIT
            ))

            # Bearing maps to horizontal position in region
            bx = int(np.clip(
                (b / (2 * np.pi) + 0.5) % 1.0 * (rw - 1), 0, rw - 1
            ))
            # Range maps to vertical position (far = top, close = bottom)
            by = int(np.clip(
                (r / self.SONIC_MAX_RANGE) * (rh - 1), 0, rh - 1
            ))
            pattern += self._gaussian_blob(rh, rw, by, bx, sigma=1.5) * amplitude

        torus.inject_region(SONIC_REGION, pattern)

    def inject_reward(self, torus: FlyBrainTorus, reward: float) -> None:
        """
        Inject reward signal as global neuromodulation.

        Positive reward → brief excitatory pulse across entire torus.
        Negative reward → brief inhibitory pulse (negative amplitude).

        Mimics dopaminergic neuromodulation in the insect mushroom body:
        reward prediction error broadcasts globally, gating plasticity
        at all active synapses simultaneously.
        """
        reward = float(reward)
        amplitude = np.clip(reward, -2.0, 2.0) * self.REWARD_AMPLITUDE
        # Broadcast uniformly across all V_in nodes
        torus.V_in += float(amplitude)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_blob(
        h: int, w: int,
        cy: int, cx: int,
        sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Generate a normalised Gaussian blob on an (h, w) grid.

        Centred at (cy, cx) with given sigma.
        Peak value = 1.0. Used to create smooth injection patterns.
        """
        yy = np.arange(h, dtype=np.float32)
        xx = np.arange(w, dtype=np.float32)
        Y, X = np.meshgrid(yy, xx, indexing='ij')
        blob = np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2.0 * sigma ** 2))
        return blob.astype(np.float32)
