# fly_brain Swarm Communication — Future Research

**Status:** Concept and theoretical framework. Not yet implemented.  
**Author:** Christian Hayes  
**Date:** April 2026  
**Repository:** github.com/ceh303-tech/fly_brain

---

## Overview

This document outlines a research direction for extending fly_brain from a
single-drone neuromorphic controller into a swarm communication architecture.
The core idea: drones communicate through continuous analogue wave signals
derived directly from their RLC torus states, with no digital protocol layer,
no packet structure, and no explicit message passing.

Swarm behaviour emerges from wave interference between coupled oscillator
networks — the same physics that drives local obstacle avoidance, extended
to inter-drone communication.

Three mechanisms work together:

1. **Differential Phasor Transmission** — efficient encoding of torus state
   changes using the phasor bus already computed each step
2. **Shot Noise Modulation** — physical security using oscillator parameters
   as cryptographic keys, transmission indistinguishable from background noise
3. **Three-Phase Implicit Error Correction** — noise rejection inherited from
   the same wave physics as local sensing, no checksum or retransmission needed

This is not a communications engineering bolt-on. It is the natural extension
of the wave physics already present in fly_brain into the radio frequency domain.

---

## Background: Why Analogue

Digital swarm communication has well-documented vulnerabilities. Packets can be
intercepted and spoofed. Protocols can be reverse-engineered. Authentication
handshakes can be replayed. GPS coordinates in a message header are an attack
surface. None of these exist in an analogue wave signal with no packet structure.

The analogue communication era (1920s-1980s) developed elegant error-correction
and security mechanisms that digital communication largely replaced with software
equivalents. Several of these map directly onto the RLC torus architecture:

| Analogue technique | Era | Maps to fly_brain via |
|-------------------|-----|----------------------|
| Spread spectrum (Lamarr 1942) | 1940s | Torus phase sequence as spreading code |
| Companded transmission (Bell Labs 1930s) | 1930s | Logarithmic torus state encoding |
| Vestigial sideband | 1940s | Previous torus state as phase reference |
| Pilot tone synchronisation | 1930s | 1/sqrt(LC) fundamental frequency |
| Walsh-Hadamard orthogonal coding | 1960s | Torus initial state as orthogonal identity |
| Differential encoding (facsimile) | 1920s | Phasor delta transmission |
| Noise carrier (stochastic) | 1950s | Shot noise modulation |
| Triplication voting | 1930s | Three-phase phasor implicit redundancy |

The three mechanisms proposed below draw from this heritage and adapt it to
the specific properties of the RLC torus.

---

## Mechanism 1: Differential Phasor Transmission

### Concept

The phasor interference bus already computes |ψ| and angle every timestep
as part of normal fly_brain operation:

```
ψ = A·exp(0) + B·exp(i·2π/3) + C·exp(i·4π/3)
```

Where A, B, C are the spike rates of the three tori (Ocellus, Halteres,
Antenna). Rather than broadcasting the full torus state — 1024 float32
values, approximately 4 KB — broadcast only the phasor delta from the
previous timestep plus a pilot tone.

### What Is Transmitted

```
broadcast packet (analogue, not digital):
  - pilot_tone:      continuous sine wave at f = 1/sqrt(L*C)
                     allows receiver to identify transmitter's RLC params
  - delta_psi_mag:   change in |ψ| from previous step (signed float)
  - delta_psi_angle: change in ψ angle from previous step (radians)
  - channel_balance: relative spike rates A:B:C (three amplitudes)
```

In analogue terms: the pilot tone is transmitted on a fixed carrier frequency.
The delta values amplitude-modulate three sub-carriers at 0°, 120°, 240°
relative to the pilot.

### Why the Torus State Changes Smoothly

The LC dynamics ensure the torus state evolves continuously. A sudden obstacle
event produces a wave that propagates over several hundred milliseconds. The
maximum rate of change of |ψ| is bounded by the LC time constant. This means:

- The delta is small relative to the absolute value in normal operation
- A corrupted delta produces a small local error that decays naturally
- No checksums needed — physics provides self-correction over time

### Prototype Implementation

```python
class PhasorTransmitter:
    """
    Computes the phasor delta from the fly_brain controller
    and prepares it for analogue transmission.
    
    On real hardware, the output would feed a DAC connected
    to an FM transmitter module. In simulation, it feeds
    directly into PhasorReceiver on other drone instances.
    
    Parameters:
        L: torus inductance (determines pilot frequency)
        C: torus capacitance (determines pilot frequency)
        signal_scale: amplitude scaling for transmission (default 1.0)
    """
    
    def __init__(self, L: float, C: float, signal_scale: float = 1.0):
        self.L = L
        self.C = C
        self.signal_scale = signal_scale
        self.pilot_freq = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
        self._prev_psi = complex(0, 0)
        self._prev_A = 0.0
        self._prev_B = 0.0
        self._prev_C = 0.0
    
    def compute_broadcast(
        self,
        spike_A: float,  # Ocellus spike rate
        spike_B: float,  # Halteres spike rate
        spike_C: float,  # Antenna spike rate
    ) -> dict:
        """
        Compute phasor and delta for this timestep.
        Returns dict suitable for transmission or simulation.
        
        In hardware: these values feed the DAC output.
        In simulation: passed directly to PhasorReceiver.
        """
        TWO_PI_3 = 2.0 * np.pi / 3.0
        
        # Compute current phasor
        psi = (spike_A * np.exp(1j * 0) +
               spike_B * np.exp(1j * TWO_PI_3) +
               spike_C * np.exp(1j * 2 * TWO_PI_3))
        
        psi_mag   = abs(psi)
        psi_angle = np.angle(psi)
        
        # Compute deltas from previous step
        prev_mag   = abs(self._prev_psi)
        prev_angle = np.angle(self._prev_psi)
        
        delta_mag   = psi_mag   - prev_mag
        delta_angle = psi_angle - prev_angle
        
        # Wrap angle delta to [-pi, pi]
        if delta_angle > np.pi:
            delta_angle -= 2 * np.pi
        if delta_angle < -np.pi:
            delta_angle += 2 * np.pi
        
        # Store for next step
        self._prev_psi = psi
        self._prev_A = spike_A
        self._prev_B = spike_B
        self._prev_C = spike_C
        
        return {
            'pilot_freq':      self.pilot_freq,
            'delta_psi_mag':   delta_mag   * self.signal_scale,
            'delta_psi_angle': delta_angle * self.signal_scale,
            'channel_A':       spike_A     * self.signal_scale,
            'channel_B':       spike_B     * self.signal_scale,
            'channel_C':       spike_C     * self.signal_scale,
            'psi_mag':         psi_mag,
            'psi_angle':       psi_angle,
        }
    
    def reset(self):
        self._prev_psi = complex(0, 0)


class PhasorReceiver:
    """
    Receives a phasor broadcast from another drone and converts
    it into torus injection voltages for the local fly_brain controller.
    
    The injection is weighted by:
    - signal_strength: proxy for distance (closer = stronger injection)
    - pilot_match: how closely the transmitter's pilot frequency matches
      the receiver's own RLC parameters (similar drones communicate better)
    
    Parameters:
        local_L, local_C: receiver's own RLC parameters (for pilot matching)
        base_injection_scale: maximum injection amplitude (default 0.3)
        pilot_tolerance: frequency tolerance for pilot matching (default 0.1)
    """
    
    def __init__(
        self,
        local_L: float,
        local_C: float,
        base_injection_scale: float = 0.3,
        pilot_tolerance: float = 0.1,
    ):
        self.local_L = local_L
        self.local_C = local_C
        self.local_pilot_freq = 1.0 / (2.0 * np.pi * np.sqrt(local_L * local_C))
        self.base_injection_scale = base_injection_scale
        self.pilot_tolerance = pilot_tolerance
        self._integrated_psi_mag = 0.0
    
    def receive(
        self,
        broadcast: dict,
        signal_strength: float,  # 0.0 (far) to 1.0 (close)
    ) -> np.ndarray:
        """
        Convert received broadcast into torus injection voltages.
        
        Returns: V_injection array of shape (N_NODES,) suitable for
        adding directly to torus.V_in before the next step.
        
        In hardware: signal_strength comes from received signal amplitude.
        In simulation: computed from inter-drone distance.
        """
        # Compute pilot match — how compatible is this transmitter
        remote_pilot = broadcast['pilot_freq']
        freq_diff = abs(remote_pilot - self.local_pilot_freq)
        freq_ratio = freq_diff / self.local_pilot_freq
        pilot_match = max(0.0, 1.0 - freq_ratio / self.pilot_tolerance)
        
        # Combined injection weight
        weight = signal_strength * pilot_match * self.base_injection_scale
        
        if weight < 1e-6:
            return np.zeros(1024, dtype=np.float32)
        
        # Integrate phasor delta
        self._integrated_psi_mag += broadcast['delta_psi_mag']
        self._integrated_psi_mag = np.clip(
            self._integrated_psi_mag, -3.0, 3.0)
        
        # Reconstruct injection pattern from channel amplitudes
        # Distribute injection across torus nodes proportional to
        # the channel balance from the transmitting drone
        import numpy as np
        N = 1024
        W = H = 32
        injection = np.zeros(N, dtype=np.float32)
        
        # Map received channel amplitudes to torus regions
        # Ocellus region: top third of torus
        # Halteres region: middle third
        # Antenna region: bottom third
        occ_nodes  = np.arange(0,     N//3)
        halt_nodes = np.arange(N//3,  2*N//3)
        ant_nodes  = np.arange(2*N//3, N)
        
        injection[occ_nodes]  = broadcast['channel_A'] * weight
        injection[halt_nodes] = broadcast['channel_B'] * weight
        injection[ant_nodes]  = broadcast['channel_C'] * weight
        
        # Modulate by integrated psi magnitude
        injection *= (1.0 + 0.2 * self._integrated_psi_mag)
        
        return injection
    
    def reset(self):
        self._integrated_psi_mag = 0.0
```

### Simulation: Two Coupled Drones

```python
def simulate_coupled_drones(n_steps: int = 1000, distance: float = 10.0):
    """
    Simulate two fly_brain controllers coupled through phasor transmission.
    
    drone_A encounters an obstacle at step 200.
    drone_B receives drone_A's phasor broadcast.
    Measure: does drone_B's torus respond before its own sensors fire?
    
    This is the swarm early-warning test.
    """
    from fly_brain.controller import FlyBrainController
    from fly_brain.simulations.sim_environment import CircularEnvironment
    
    # Create two independent drones
    drone_A = FlyBrainController()
    drone_B = FlyBrainController()
    
    env_A = CircularEnvironment(rng_seed=42)
    env_B = CircularEnvironment(rng_seed=99)
    
    # Create communication channel
    tx_A = PhasorTransmitter(L=0.01, C=0.01)
    rx_B = PhasorReceiver(local_L=0.01, local_C=0.01)
    
    # Signal strength decays with distance (inverse square law)
    MAX_RANGE = 50.0
    signal_strength = max(0.0, 1.0 - (distance / MAX_RANGE) ** 2)
    
    results = {
        'A_spike_rate': [],
        'B_spike_rate_own':  [],
        'B_spike_rate_with_swarm': [],
        'B_energy_own': [],
        'B_energy_with_swarm': [],
    }
    
    drone_B_swarm = FlyBrainController()  # B with swarm input
    rx_B2 = PhasorReceiver(local_L=0.01, local_C=0.01)
    
    for step in range(n_steps):
        obs_A = env_A.step()
        obs_B = env_B.step()
        
        # Step drone A
        motors_A = drone_A.step(
            accel=obs_A['accel'], gyro=obs_A['gyro'],
            ranges=obs_A['ranges'], bearings=obs_A['bearings'])
        
        # Compute A's broadcast
        sr_occ  = drone_A.torus_ocellus.spike_rate()
        sr_halt = drone_A.torus_halteres.spike_rate()
        sr_ant  = drone_A.torus_antenna.spike_rate()
        broadcast = tx_A.compute_broadcast(sr_occ, sr_halt, sr_ant)
        
        # Step drone B without swarm input
        motors_B = drone_B.step(
            accel=obs_B['accel'], gyro=obs_B['gyro'],
            ranges=obs_B['ranges'], bearings=obs_B['bearings'])
        
        # Step drone B WITH swarm input from A
        swarm_injection = rx_B2.receive(broadcast, signal_strength)
        drone_B_swarm.torus_ocellus.V_in  += swarm_injection[:341]
        drone_B_swarm.torus_halteres.V_in += swarm_injection[341:682]
        drone_B_swarm.torus_antenna.V_in  += swarm_injection[682:]
        motors_B_swarm = drone_B_swarm.step(
            accel=obs_B['accel'], gyro=obs_B['gyro'],
            ranges=obs_B['ranges'], bearings=obs_B['bearings'])
        
        # Record
        results['A_spike_rate'].append(drone_A.torus_ocellus.spike_rate())
        results['B_spike_rate_own'].append(drone_B.torus_ocellus.spike_rate())
        results['B_spike_rate_with_swarm'].append(
            drone_B_swarm.torus_ocellus.spike_rate())
        results['B_energy_own'].append(drone_B.torus_ocellus.mean_energy())
        results['B_energy_with_swarm'].append(
            drone_B_swarm.torus_ocellus.mean_energy())
    
    return results


def analyse_early_warning(results: dict) -> dict:
    """
    Measure whether swarm coupling provides early warning.
    
    Find the step where drone A's spike rate first exceeds 0.05
    (obstacle detected). Find the step where B's torus first responds
    with and without swarm input. The difference is the early warning time.
    """
    import numpy as np
    
    A_sr  = np.array(results['A_spike_rate'])
    B_own = np.array(results['B_spike_rate_own'])
    B_sw  = np.array(results['B_spike_rate_with_swarm'])
    
    A_detect = np.where(A_sr > 0.05)[0]
    B_own_detect = np.where(B_own > 0.05)[0]
    B_sw_detect  = np.where(B_sw  > 0.05)[0]
    
    A_step = A_detect[0] if len(A_detect) > 0 else None
    B_own_step = B_own_detect[0] if len(B_own_detect) > 0 else None
    B_sw_step  = B_sw_detect[0]  if len(B_sw_detect)  > 0 else None
    
    early_warning_steps = None
    if B_own_step is not None and B_sw_step is not None:
        early_warning_steps = B_own_step - B_sw_step
    
    return {
        'A_detects_at_step': A_step,
        'B_detects_at_step_alone': B_own_step,
        'B_detects_at_step_swarm': B_sw_step,
        'early_warning_steps': early_warning_steps,
        'early_warning_ms': (early_warning_steps * 10)
            if early_warning_steps else None,
    }
```

---

## Mechanism 2: Shot Noise Modulation

### Concept

Each drone's torus generates current-scaled shot noise every step:

```python
I += 0.002 * np.sqrt(np.mean(np.abs(I))) * rng.standard_normal(N)
```

The statistical distribution of this noise depends on the current state of
the torus — specifically on mean(|I|), which changes with the wave dynamics.
The noise is not random in the sense of being unpredictable to the drone
itself. Given the RLC parameters L and C, the noise distribution at any
moment is deterministic given the wave state.

Shot noise modulation exploits this. The transmitting drone slightly varies
its effective inductance L in a pattern derived from the wave state it wants
to transmit. This shifts the shot noise statistics. A receiving drone that
knows the transmitter's L and C values can detect the statistical shift and
reconstruct the transmitted wave state.

To any other observer the signal is indistinguishable from background noise.
There is no carrier frequency to detect, no modulation pattern to analyse,
no structure that reveals the presence of a communication signal. The L and C
values are the encryption key.

### Why This Is Physically Motivated

Real tunnel diodes exhibit current noise that depends on operating point
through the Schottky formula:

```
S_I = 2 * e * I_dc    (current noise power spectral density)
```

Where e is electron charge and I_dc is the DC bias current. The noise is not
constant — it scales with current. Your current-scaled shot noise model
implements exactly this relationship.

Varying the effective L changes the resonant frequency, which changes the
current distribution, which changes the noise statistics according to the
Schottky formula. The modulation is physically motivated at the device level.

### Implementation

```python
class ShotNoiseModulator:
    """
    Modulates the RLC torus shot noise to carry a communication signal.
    
    The modulation is applied by varying the effective inductance L
    in the torus step function. The variation is small (default 2%)
    to avoid perturbing the wave dynamics significantly.
    
    Security: the L and C values of the transmitter are the
    cryptographic key. Without them, the transmitted signal is
    statistically indistinguishable from background noise.
    
    Parameters:
        L_nominal:        torus nominal inductance
        C:                torus capacitance
        modulation_depth: fraction of L to modulate (default 0.02 = 2%)
        carrier_symbol:   number of torus steps per transmitted symbol
    """
    
    def __init__(
        self,
        L_nominal: float,
        C: float,
        modulation_depth: float = 0.02,
        carrier_symbol: int = 10,
    ):
        self.L_nominal = L_nominal
        self.C = C
        self.modulation_depth = modulation_depth
        self.carrier_symbol = carrier_symbol
        self._symbol_buffer = []
        self._step = 0
    
    def encode_psi_as_symbols(self, psi_mag: float, psi_angle: float,
                               n_bits: int = 4) -> list:
        """
        Convert psi magnitude and angle into a sequence of L modulation values.
        
        Quantises psi_mag to n_bits levels and encodes as a sequence of
        small L variations. The sequence is the transmitted symbol.
        
        In hardware: these L values would be applied to a varactor diode
        (voltage-controlled capacitor) in the RLC circuit.
        In simulation: applied as a parameter variation in the torus step.
        """
        levels = 2 ** n_bits
        
        # Normalise to [0, levels-1]
        psi_quantised = int(np.clip(psi_mag * levels, 0, levels - 1))
        angle_quantised = int(np.clip(
            (psi_angle + np.pi) / (2 * np.pi) * levels, 0, levels - 1))
        
        # Encode as L variation sequence
        # Each bit maps to a small positive or negative L deviation
        symbols = []
        for bit_idx in range(n_bits):
            psi_bit   = (psi_quantised   >> bit_idx) & 1
            angle_bit = (angle_quantised >> bit_idx) & 1
            
            # Differential encoding: 1 = increase L, 0 = decrease L
            delta_L_psi   = self.L_nominal * self.modulation_depth * (1 if psi_bit   else -1)
            delta_L_angle = self.L_nominal * self.modulation_depth * (1 if angle_bit else -1)
            
            symbols.append({
                'L_psi':   self.L_nominal + delta_L_psi,
                'L_angle': self.L_nominal + delta_L_angle,
            })
        
        return symbols
    
    def get_current_L(self, psi_mag: float, psi_angle: float) -> float:
        """
        Return the effective L value for this timestep.
        
        Called every torus step. Returns a slightly varied L that
        encodes the current psi state in the shot noise statistics.
        """
        symbols = self.encode_psi_as_symbols(psi_mag, psi_angle)
        symbol_idx = (self._step // self.carrier_symbol) % len(symbols)
        bit_idx    = self._step % 2  # alternate between psi and angle bits
        
        self._step += 1
        
        if bit_idx == 0:
            return symbols[symbol_idx]['L_psi']
        else:
            return symbols[symbol_idx]['L_angle']


class ShotNoiseDemodulator:
    """
    Detects shot noise modulation from a received signal and
    reconstructs the transmitting drone's psi state.
    
    Monitors the statistical distribution of incoming signal noise.
    Detects deviations from the expected distribution given the
    receiver's own L and C parameters.
    
    Parameters:
        L_key:              transmitter's L value (the cryptographic key)
        C_key:              transmitter's C value
        window_steps:       number of steps to average over (default 20)
        detection_threshold: minimum statistical deviation to detect (default 0.01)
    """
    
    def __init__(
        self,
        L_key: float,
        C_key: float,
        window_steps: int = 20,
        detection_threshold: float = 0.01,
    ):
        self.L_key = L_key
        self.C_key = C_key
        self.window_steps = window_steps
        self.detection_threshold = detection_threshold
        self._noise_buffer = []
        self._decoded_psi_history = []
    
    def observe(self, received_noise_sample: float) -> dict:
        """
        Record one received noise sample and attempt to decode.
        
        received_noise_sample: the amplitude of the incoming signal
        at this timestep (from radio receiver in hardware, or
        directly from the transmitting torus in simulation).
        
        Returns decoded psi estimate when enough samples accumulated,
        otherwise returns None.
        """
        self._noise_buffer.append(received_noise_sample)
        
        if len(self._noise_buffer) < self.window_steps:
            return {'decoded': False, 'psi_mag': None, 'psi_angle': None}
        
        # Keep only recent window
        if len(self._noise_buffer) > self.window_steps * 2:
            self._noise_buffer = self._noise_buffer[-self.window_steps:]
        
        buffer = np.array(self._noise_buffer[-self.window_steps:])
        
        # Compute observed noise statistics
        observed_mean = np.mean(np.abs(buffer))
        observed_std  = np.std(buffer)
        
        # Expected statistics from nominal L and C (no modulation)
        # Under Schottky formula, noise variance scales with current
        # At rest, I ~ 0, so expected noise is near zero
        # With modulation, noise statistics shift predictably
        expected_std_nominal = 0.002  # nominal shot noise amplitude
        
        # Deviation from expected indicates modulation
        deviation = (observed_std - expected_std_nominal) / expected_std_nominal
        
        if abs(deviation) < self.detection_threshold:
            return {'decoded': False, 'psi_mag': None, 'psi_angle': None}
        
        # Reconstruct psi from deviation pattern
        # Positive deviation = higher L symbol = bit 1
        # Negative deviation = lower L symbol = bit 0
        psi_mag_estimate   = np.clip(abs(deviation) * 5.0, 0.0, 1.0)
        psi_angle_estimate = np.arctan2(
            np.mean(buffer[len(buffer)//2:]),
            np.mean(buffer[:len(buffer)//2])
        )
        
        result = {
            'decoded':    True,
            'psi_mag':    psi_mag_estimate,
            'psi_angle':  psi_angle_estimate,
            'confidence': min(1.0, abs(deviation) / 0.1),
            'deviation':  deviation,
        }
        
        self._decoded_psi_history.append(result)
        return result
```

### Security Analysis

An adversary observing the channel sees:

- A signal with variance approximately 0.002 x sqrt(mean|I|)
- No detectable carrier frequency
- No periodic structure in the signal
- Statistical properties matching background noise

To spoof a signal the adversary must:

1. Know the transmitter's L and C values (the key)
2. Know the current torus wave state (changes 100 times per second)
3. Reproduce the exact shot noise modulation pattern

Without L and C, step 1 is impossible from the observed signal alone.
Even with L and C, reproducing the current wave state requires running
the exact same RLC simulation with the same sensor inputs as the target
drone — effectively impossible without access to the drone hardware.

This is closer to a physical unclonable function than to conventional
cryptography. The security derives from physical parameters, not
mathematical assumptions about computational hardness.

---

## Mechanism 3: Three-Phase Implicit Error Correction

### Concept

Each drone transmits three analogue signals phase-locked to its three tori
at 0°, 120°, and 240°. The receiving drone computes the phasor sum of the
incoming three-phase signal exactly as it would for its own three internal tori.

A corrupted or interfered channel produces a phasor component that is
incoherent with the other two. Rather than corrupting the decoded signal,
the incoherent component reduces |ψ| — it partially cancels rather than
adds noise. The error degrades gracefully rather than catastrophically.

This is the majority voting principle implemented at the wave physics level.
No explicit error detection. No retransmission. The physics absorbs the error.

### Why 120° Separation

At 120° separation the three phasors are maximally orthogonal. A corrupted
channel adds a random-phase component to the sum. The expected value of a
random-phase unit phasor added to two coherent unit phasors is:

```
|ψ_corrupted| = sqrt(2/3) * |ψ_clean|  ≈ 0.82 * |ψ_clean|
```

The signal degrades by 18% when one of three channels is completely lost.
This is deterministic and predictable — the receiving drone knows that a
reduced |ψ| may indicate a corrupted channel rather than a genuine decrease
in the transmitted psi magnitude.

At any other phase separation the degradation would be asymmetric —
some channel combinations would degrade more than others, making the
corruption harder to characterise.

### Channel Quality Monitoring

```python
class ThreePhaseChannelMonitor:
    """
    Monitors the quality of the three-phase swarm communication channel.
    
    Detects corrupted channels by measuring coherence between the
    three received signals. A coherent three-phase signal produces
    |psi| close to the expected value. Incoherence from a corrupted
    channel produces reduced |psi| in a characteristic way.
    
    Parameters:
        window_steps:         averaging window for coherence measurement
        corruption_threshold: |psi| reduction that indicates corruption
    """
    
    def __init__(self, window_steps: int = 50,
                 corruption_threshold: float = 0.15):
        self.window_steps = window_steps
        self.corruption_threshold = corruption_threshold
        self._psi_history = []
        self._channel_history = {'A': [], 'B': [], 'C': []}
    
    def update(self, ch_A: float, ch_B: float, ch_C: float) -> dict:
        """
        Record received channel amplitudes and assess channel health.
        
        ch_A, ch_B, ch_C: received amplitudes from three phase channels.
        
        Returns dict with:
            psi_mag:          magnitude of received phasor
            psi_angle:        angle of received phasor
            channel_coherence: 0.0 (fully corrupted) to 1.0 (clean)
            suspect_channel:  which channel appears corrupted (A, B, C, or None)
            correction_applied: whether psi was corrected using two clean channels
        """
        TWO_PI_3 = 2.0 * np.pi / 3.0
        
        # Compute phasor from received channels
        psi = (ch_A * np.exp(1j * 0) +
               ch_B * np.exp(1j * TWO_PI_3) +
               ch_C * np.exp(1j * 2 * TWO_PI_3))
        
        psi_mag   = abs(psi)
        psi_angle = np.angle(psi)
        
        self._psi_history.append(psi_mag)
        self._channel_history['A'].append(ch_A)
        self._channel_history['B'].append(ch_B)
        self._channel_history['C'].append(ch_C)
        
        if len(self._psi_history) < self.window_steps:
            return {
                'psi_mag': psi_mag,
                'psi_angle': psi_angle,
                'channel_coherence': 1.0,
                'suspect_channel': None,
                'correction_applied': False,
            }
        
        # Keep only recent window
        self._psi_history = self._psi_history[-self.window_steps:]
        for ch in 'ABC':
            self._channel_history[ch] = (
                self._channel_history[ch][-self.window_steps:])
        
        # Check channel coherence
        # Expected: all three channels should have similar variance
        # A corrupted channel will have anomalous variance
        variances = {
            'A': np.var(self._channel_history['A']),
            'B': np.var(self._channel_history['B']),
            'C': np.var(self._channel_history['C']),
        }
        
        mean_var = np.mean(list(variances.values()))
        std_var  = np.std(list(variances.values()))
        
        suspect = None
        correction_applied = False
        corrected_psi_mag = psi_mag
        
        if std_var > mean_var * self.corruption_threshold and mean_var > 1e-6:
            # One channel has anomalous variance — likely corrupted
            suspect = max(variances, key=variances.get)
            
            # Reconstruct psi from the two clean channels
            clean_channels = {k: v for k, v in variances.items()
                              if k != suspect}
            ch_vals = {
                'A': (ch_A, 0),
                'B': (ch_B, TWO_PI_3),
                'C': (ch_C, 2 * TWO_PI_3),
            }
            
            psi_corrected = sum(
                ch_vals[k][0] * np.exp(1j * ch_vals[k][1])
                for k in clean_channels
            )
            # Scale up to compensate for missing channel
            # (two channels of three gives 2/3 of the signal)
            psi_corrected *= 1.5
            corrected_psi_mag = abs(psi_corrected)
            psi_angle = np.angle(psi_corrected)
            correction_applied = True
        
        # Channel coherence metric (1.0 = fully coherent, 0.0 = incoherent)
        coherence = 1.0 - min(1.0, std_var / (mean_var + 1e-9))
        
        return {
            'psi_mag':            corrected_psi_mag,
            'psi_angle':          psi_angle,
            'channel_coherence':  coherence,
            'suspect_channel':    suspect,
            'correction_applied': correction_applied,
            'raw_psi_mag':        psi_mag,
        }
```

---

## Swarm Topology and Range

The natural communication topology is distance-limited local coupling.
Signal strength decays with distance. Beyond a maximum range the signal
falls below the noise floor and the coupling vanishes.

This produces emergent flocking behaviour without any explicit coordination:

- Close drones share strong wave coupling — their tori partially synchronise
- Distant drones are independent — no coupling, no influence
- Medium-range drones receive a weak coupling that biases their wave state

The swarm has no fixed topology, no leader, no routing protocol. The
topology is defined entirely by the physics of signal propagation.

### Estimated Ranges with Low-Power Radio

Using an FM transmitter module (Si4713, about £3) at maximum 0.5W output:

| Environment | Estimated range | Coupling strength |
|-------------|----------------|------------------|
| Open field | 200-500m | Strong |
| Urban | 50-150m | Moderate |
| Indoor | 10-30m | Weak |
| Through obstacles | 5-15m | Very weak |

At these ranges a swarm of 5-10 drones could maintain continuous analogue
wave coupling across a typical search-and-rescue or agricultural survey area.

### Signal Attenuation Model

```python
def signal_strength(distance: float, max_range: float = 200.0,
                    exponent: float = 2.0) -> float:
    """
    Compute signal strength as a function of distance.
    
    Uses inverse power law. Exponent=2.0 for free space.
    Exponent=3.0-4.0 for urban environments.
    
    Returns float in [0.0, 1.0]. Below a minimum threshold
    (distance > max_range) returns 0.0 — below noise floor.
    """
    if distance >= max_range:
        return 0.0
    return max(0.0, (1.0 - distance / max_range) ** exponent)
```

---

## Emergent Swarm Behaviours

The following behaviours should emerge from wave coupling alone,
without any explicit swarm coordination logic.

### Obstacle Propagation (Early Warning)

When drone A detects an obstacle, its torus state changes.
This propagates to nearby drone B via phasor transmission.
B's torus develops a weaker version of the same pattern before
B's own sensors detect anything.

The early warning distance is approximately:

```
early_warning_range = swarm_separation * (wave_memory_ms / sensor_latency_ms)
```

With wave memory of 2000ms and sensor latency of 33ms (one IMU frame):
early warning extends approximately 60 drone-separations through the swarm.

### Formation Keeping

Two drones flying in formation develop correlated torus states —
similar obstacle patterns, similar motion history. When one deviates,
their torus states decorrelate. The other drone receives a phasor signal
indicating decorrelation (|ψ| drops) and adjusts behaviour to restore
correlation. Formation keeping emerges from wave coherence without any
explicit position tracking.

### Collective Avoidance

A flock of birds encountering a predator produces a coherent evasion wave
that propagates through the flock faster than each individual bird could
react to the predator alone. This is wave communication — the evasion
pattern propagates as a wave through the coupled oscillator network of
birds, each coupling to its neighbours.

The fly_brain swarm implements exactly this. An obstacle encountered by
one drone propagates as a wave state change through the swarm via phasor
transmission. The swarm reacts as a collective faster than individual
sensor latency would allow.

---

## Hardware Implementation Path

### Minimum Hardware per Drone (Additional to Existing BOM)

| Component | Purpose | Cost |
|-----------|---------|------|
| MCP4922 dual DAC | Convert torus phasor values to analogue voltage | £3 |
| Si4713 FM transmitter | Broadcast analogue signal | £3 |
| Si4735 FM receiver | Receive other drones' signals | £3 |
| BB112 varactor diode (x3) | Voltage-controlled capacitance for shot noise modulation | £0.50 |
| Shielded coax (1m) | Low-noise connection from Pi to transmitter | £2 |
| **Total additional per drone** | | **~£12** |

Total hardware cost per drone including existing dronewarp and fly_brain BOM:
approximately £100.

### Wiring Notes

The radio transmitter and receiver must be physically separated from the
torus analogue circuitry (if implemented in hardware) to avoid the wiring
noise problem identified in the wave_vision testing. The 0.1 mV per node
wiring noise budget applies to the torus input lines. The radio section
can tolerate much higher noise levels on its own lines.

### Simulation Before Hardware

All three mechanisms can be simulated in software by running multiple
FlyBrainController instances in the same Python process and passing
broadcast dicts between them. No radio hardware required to validate
the swarm dynamics. The hardware is needed only to validate the
analogue transmission medium.

Recommended simulation path:
1. Two coupled drones, circular environments, measure early warning
2. Three to five coupled drones, measure formation coherence
3. Ten drones, measure collective avoidance against a moving obstacle

---

## Open Research Questions

These are the questions that a paper on fly_brain swarm communication
would need to answer. They are not answered here — this document is a
research agenda, not a completed study.

**Q1 — Early warning quantification**
How many timesteps of early warning does a swarm drone receive before
its own sensors detect an obstacle that a neighbouring drone has already
seen? How does this scale with swarm density and inter-drone distance?

**Q2 — Formation coherence threshold**
At what phasor correlation level do two drones begin to naturally fly
in formation? Is there a critical coupling strength below which formation
is unstable?

**Q3 — Shot noise demodulation sensitivity**
What is the minimum detectable modulation depth for shot noise
demodulation? What is the effective data rate? What signal-to-noise
ratio is required for reliable decoding?

**Q4 — Swarm resilience**
How does the collective behaviour degrade as drones fail or leave range?
Is there a critical connectivity threshold below which swarm coherence
collapses?

**Q5 — Adversarial robustness**
Can a spoofed analogue signal fool the swarm? What power level is needed
to overwhelm the shot noise carrier? Does the GABA lateral inhibition
naturally reject incoherent spoofed signals?

**Q6 — Hebbian swarm learning**
Over many flights, do the Hebbian weights in each drone adapt to the
typical communication patterns of its swarm neighbours, effectively
learning each other's wave signatures? If so, does this improve
coupling efficiency over time?

---

## Relationship to Existing Research

### Coupled Oscillator Swarms

The Kuramoto model describes synchronisation in networks of coupled
oscillators. Most implementations use mathematical oscillators with
no physical substrate. fly_brain uses physical RLC oscillators with
richer dynamics — negative resistance, shot noise, GABA inhibition.

The fly_brain swarm extends Kuramoto-style coupling to include:
- Asymmetric coupling (stronger for similar L/C drones)
- Information content in the phase relationship (phasor angle encodes
  dominant obstacle direction)
- Noise-mediated coupling (shot noise modulation)
- Three-phase implicit error correction

### Stigmergy

Stigmergy — coordination through environment modification — is used by
ants and termites. Pheromone gradients are modified by one agent and
sensed by others, producing coordinated behaviour with no direct
communication.

The fly_brain swarm is electromagnetic stigmergy. Each drone modifies
the radio frequency environment through its broadcast. Other drones
sense this modification through their receivers. The swarm coordinates
without direct communication — only indirect coupling through the
shared electromagnetic medium.

### Neuromorphic Swarms

Neuromorphic computing applied to swarm robotics is an emerging field.
Published work (as of 2026) primarily uses Intel Loihi or IBM TrueNorth
as the onboard processor with standard digital radio communication.

The fly_brain swarm architecture differs in using:
- Analogue communication derived from the neuromorphic computation itself
- Shot noise as a communication medium (physically motivated by the
  same tunnel diode physics that drives local computation)
- No separation between computation and communication substrates

This is the first proposed architecture where the communication medium
is the same physical phenomenon — RLC oscillator wave dynamics — as
the computation medium.

---

## Next Steps

**Immediate (simulation):**
- Implement PhasorTransmitter and PhasorReceiver in fly_brain/swarm/
- Run two-drone coupling simulation
- Measure early warning timesteps across a range of inter-drone distances
- Validate that wave coupling does not destabilise individual drone behaviour

**Short term (hardware):**
- Add DAC and FM transmitter to one Pi Zero 2W
- Verify analogue phasor signal integrity on oscilloscope
- Two-drone ground test with physical radio link
- Measure actual signal attenuation versus distance

**Medium term (swarm):**
- Scale to five drones
- Test collective obstacle avoidance
- Measure formation coherence with and without swarm coupling
- Compare reaction time to obstacle as function of swarm density

**Publication target:**
The most publishable result is a demonstration that a swarm of
fly_brain drones exhibits faster collective obstacle avoidance than
any individual drone could achieve with its own sensors alone,
using analogue wave coupling with no digital protocol.

Appropriate venues: IEEE Transactions on Cognitive and Developmental
Systems, Swarm Intelligence journal, or the International Conference
on Neuromorphic Systems (ICONS).

---

## Files to Create

When implementing, the following file structure is recommended:

```
fly_brain/
|-- swarm/
|   |-- __init__.py
|   |-- phasor_transmitter.py   (PhasorTransmitter class)
|   |-- phasor_receiver.py      (PhasorReceiver class)
|   |-- shot_noise_modulator.py (ShotNoiseModulator, ShotNoiseDemodulator)
|   |-- channel_monitor.py      (ThreePhaseChannelMonitor)
|   |-- signal_model.py         (signal_strength, attenuation models)
|   +-- swarm_controller.py     (multi-drone simulation runner)
|-- simulations/
|   +-- test_swarm_coupling.py  (early warning benchmark)
+-- docs/
    +-- SWARM_RESEARCH.md       (this document)
```

---

## Author Note

The three communication mechanisms described here emerged from thinking
about what is already present in fly_brain and asking what happens if
you extend it outward from one drone to many.

The shot noise is already there — it drives stochastic resonance in the
local sensors. Modulating it for communication is the natural extension.
The phasor bus is already there — it couples the three local tori. Broadcasting
it to other drones is the natural extension. The three-phase error correction
is already there — it provides implicit redundancy in local visual encoding.
Applying it to the radio channel is the natural extension.

None of these required importing an idea from outside the architecture.
They are all already present in the fly_brain physics, extended in the
obvious direction.

The analogue era communication techniques provide the mathematical
framework for understanding and analysing what the physics produces
naturally. Spread spectrum, companding, vestigial sideband, Walsh coding —
these are not new ideas being grafted onto the architecture. They are the
names that engineers gave to physical phenomena that the RLC torus exhibits
already.

---

*Christian Hayes — Independent aerospace researcher — April 2026*  
*github.com/ceh303-tech/fly_brain*  
*MIT Licence — Free for student and research use*
