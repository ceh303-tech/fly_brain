# fly_brain

**A wave-based neuromorphic drone navigation controller inspired by the Drosophila optic lobe and bat echolocation.**

No LLM. No backpropagation. No GPU. No pretrained weights.
Just wave physics and Hebbian learning, running in roughly 140 KB on a Raspberry Pi Zero 2W (62 KB at float16).

---

## The Core Idea

Most drone AI systems process the world as a sequence of snapshots. Take a frame, run inference, output a decision, repeat. Each inference is independent. There is no memory between frames except what is visible in the current image.

A fly doesn't work like that. A fly's nervous system is continuously resonating with its environment. There is no discrete inference step. The wave dynamics across its optic lobe are the cognition; perception and action are the same process, not a pipeline.

fly_brain is an attempt to build a drone controller that works the same way. The wave state across a toroidal grid of oscillator nodes is the world model. Motor commands emerge from that wave state continuously. The network learns by changing the connections between nodes as it accumulates flight experience, with no training dataset and no offline learning phase.

---

## Why It Was Designed This Way: A Note on Dyscalculia

I have dyscalculia. Numbers don't naturally arrange themselves into meaningful patterns for me the way they do for most people. A page of differential equations is largely illegible without significant effort. Matrix algebra written as symbols on a page tells me almost nothing.

The way I got around this was to stop trying to read the equations and start trying to visualise what was physically happening. A neuron is a physical object. It has a membrane. That membrane has electrical properties. You can build a circuit that behaves the same way, and once you have a circuit you can reason about it using things you can see and touch. Capacitors store charge. Inductors resist changes in current. Resistors dissipate energy. A tunnel diode pumps energy back in. These are all things I can draw, simulate, and physically build.

The RLC oscillator neuron model in fly_brain came directly from this approach. Rather than starting from the Hodgkin-Huxley equations and simplifying, I started from the question: what circuit would a neuron be if you had to build one on a breadboard? The answer turned out to be an RLC tank circuit with a tunnel diode, which when you go back and look at the Hodgkin-Huxley model is actually a reasonable approximation of what the membrane equations describe. I got to the right answer from the wrong direction.

The same approach applies to the network architecture. I couldn't visualise a high-dimensional weight matrix updating via backpropagation. But I could visualise a wave travelling across a surface, bouncing off boundaries, interfering with other waves, and gradually carving channels through a medium as it repeatedly travels the same paths. That is Hebbian learning on a torus, described as something you can watch rather than only calculate.

I use AI assistance extensively for the arithmetic and matrix algebra (a registered reasonable adjustment for dyscalculia, disclosed in all academic work). The architecture, biological analogies, and design decisions are my own.

Whether the circuit analogy produces a better architecture than starting from the equations is an open question. What it produced is an architecture that no one else seems to have built, which is at least interesting.

---

## Why Waves Instead of Numbers

Standard artificial neurons compute a weighted sum and pass it through an activation function. The output is a number. The neuron resets immediately. Nothing persists between forward passes.

Real neurons are physical oscillators. A neuron has a membrane capacitance that stores charge, inductive ion channel dynamics, and a resistance representing leak conductance. It oscillates. It rings after stimulation. It carries history forward in its physical state rather than in an explicit memory register.

When many neurons are connected and one fires, a wave propagates outward through the network. That wave interferes with waves from other sources. The interference pattern encodes relationships between events that no individual neuron could represent alone. A wave pattern has phase, frequency, amplitude, and direction. Binary numbers have none of these properties.

This is fundamentally richer than passing numbers through layers. The brain of a fly contains roughly 100,000 neurons, most of them dedicated to processing visual motion as wave interference across the optic lobe. A fly detects, decides, and reacts in about 30 milliseconds. The architecture that achieves this is not deep learning. fly_brain is an attempt to understand why, and to build something that works on the same principles.

---

## How It All Fits Together: Data Flow

```
IMU (100 Hz)                    Ultrasonic sensors
accel + gyro                    range + bearing to obstacles
     |                                   |
     v                                   v
SensorInjector.inject_imu()    SensorInjector.inject_ultrasonic()
     |                                   |
     |  voltage injected at              |  amplitude proportional to 1/range
     |  IMU tori regions                 |  position proportional to bearing
     |                                   |
     v                                   v
+-------------------------------------------------------------+
|              THREE TORI  (each 32x32 = 1024 nodes)          |
|                                                             |
|  OCELLUS torus   : clearance (how far obstacles are)        |
|  HALTERES torus  : angular velocity / rotation              |
|  ANTENNA torus   : proximity + linear acceleration          |
|                                                             |
|  Each torus:                                                |
|  * 1024 RLC oscillator nodes                                |
|  * 4 sub-neurons per node (local microcircuit)              |
|  * Tunnel diode: shot noise + negative resistance           |
|  * GABA: divisive normalisation + lateral inhibition        |
|  * Hebbian plasticity on 8-neighbour sparse connections     |
|  * Impedance boundaries: L x 1.5/1.3, partial reflection   |
+-------------------------------------------------------------+
     |  spike states from each torus
     |
     |  Lobula Plate Interference Bus (120 degree phase separation)
     |  psi = A*exp(i*0) + B*exp(i*2pi/3) + C*exp(i*4pi/3)
     |  |psi| large when one phase dominates (novel event)
     |  |psi| near zero when all three balance (background)
     |  feeds back to all three tori as shared resonance
     |
     |  Balanced ternary readout:
     |  trit = V_ocellus - V_antenna
     |  trit > 0  =>  safe   (reward +1)
     |  trit < 0  =>  danger (reward -1)
     |
     v
MotorReadout (linear W: N -> 4 motors)
Trained online via delta rule + reward signal
EWC anchoring prevents overwriting on environment change
     |
     v
Motor commands [0.0 to 1.0] x 4 rotors
     |
     v
Hebbian update: connections that fired together strengthen
Replay buffer samples past injection patterns every 20 steps
     |
     v
Next timestep (dt = 0.01 s, 100 Hz)
```

---

## The Neuron Model: RLC Resonator with Tunnel Diode

Each node is modelled as a physical RLC circuit, a resistor, inductor, and capacitor in series, with a tunnel diode in parallel.

```
        R (resistance)      L (inductance)
*---[resistor]---[inductor]---*
                               |
                            C (capacitance)
                               |
                          [Tunnel Diode]
                               |
                              GND
```

**The capacitor** stores charge (membrane capacitance). Voltage across it is the membrane potential.

**The inductor** creates oscillation. When current is interrupted the inductor generates a back-EMF that keeps current flowing. This makes the circuit ring at its natural frequency 1/sqrt(LC) after stimulation rather than simply decaying. A pure RC circuit just decays. Adding inductance creates genuine oscillatory dynamics.

**The resistor** provides damping and controls how long the oscillation persists. High resistance means fast decay; low resistance means the node rings for longer.

**The tunnel diode does two jobs simultaneously.**

*Shot noise.* Current via quantum mechanical tunnelling is discrete and probabilistic. Each tunnelling electron is a random impulse producing shot noise (Schottky formula) injected into the circuit at every timestep. Small amounts of noise help the node detect sub-threshold signals it would otherwise miss; this is stochastic resonance. Flies use ion channel stochasticity the same way to enhance sensitivity to weak visual motion.

*Negative resistance.* A tunnel diode has a region of its current-voltage curve where increasing voltage decreases current. This negative resistance pumps energy into the circuit rather than dissipating it. Combined with the LC tank circuit, it compensates for resistive losses and sustains oscillation after external stimulation ends. After an obstacle echo injects energy into the antenna torus, nodes in that region keep oscillating for approximately 2000 ms. The drone remembers where the obstacle was as it flies past.

This is the same physics as the Soviet GaAs tunnel diodes used in QRNG hardware, which sustain microwave oscillations through negative resistance. The same physical effect, applied to neural computation.

---

## Nodes Inside Nodes: The Sub-Neuron Architecture

Each position on the 32x32 grid is a **node**. Inside each node are **4 sub-neurons**. Each sub-neuron is one complete RLC oscillator.

A real cortical column contains hundreds of neurons densely connected internally. Before any signal leaves the column to travel to distant regions, it has been processed within the column first. The sub-neurons are that local microcircuit; the node is the column.

Each sub-neuron responds to an incoming wave at a slightly different phase. The output that propagates to neighbouring nodes is a superposition of four slightly different oscillations rather than a single clean sine wave, which produces richer interference patterns, more discriminable states, and faster learning.

In a fly's optic lobe, each pixel of the visual field maps to a cartridge of approximately six neurons that process that pixel locally before projecting outward. The node is the cartridge. The sub-neurons are those six neurons, simplified to four.

---

## Three Tori: Drosophila Sensory Modalities

fly_brain uses three separate 32x32 tori at 120 degree phase separation, each mirroring a distinct Drosophila sensory pathway:

| Torus | Biological analogue | Encodes | Trit pole |
|-------|---------------------|---------|-----------|
| **Ocellus** | Dorsal ocelli (simple eyes) | Clearance (how far obstacles are) | +1 |
| **Halteres** | Modified hindwings (gyroscopes) | Angular velocity and rotation | 0 |
| **Antenna** | Compound eye and Johnston's organ | Proximity and linear acceleration | -1 |

The three tori couple through a 120 degree-separated interference bus, the lobula plate tangential cells (LPTCs) of the fly optic lobe. The balanced ternary readout computes trit = V_ocellus minus V_antenna, which is directly correlated with reward and converges in tens of steps without pre-training.

The 120 degree spacing came from thinking about three-phase AC power. Three-phase mains uses 120 degree offsets between phases because that is the only angle that divides a full circle into three equal symmetric sectors. At any other spacing one pair of phases would be closer together than the others and their wave patterns would partially correlate. At 120 degrees you get maximum mutual orthogonality; the three phases are as different from each other as it is physically possible to be.

The same logic applies here. If all three tori ran at the same phase their wave patterns would be indistinguishable when the readout tries to work out which signal came from where. Giving each torus a different phase offset is the same as giving each AC line in a power grid a unique electrical fingerprint. An obstacle event and a rotation event produce physically distinct spatial patterns on their respective tori even when the raw signal amplitude is identical.

The cross-phase coupling also exploits this directly. When one torus is near its zero-crossing, a small amount of its noise is injected into the other two tori as stochastic resonance, helping them detect sub-threshold signals they would otherwise miss. This only works cleanly because the three phases are always at different points in their oscillation cycle. At 120 degree spacing you get maximum independence between the tori and maximum stochastic resonance benefit simultaneously.

This turned out to match the T4 and T5 direction-selective motion cells in a real fly's medulla, which process the visual field from three arm directions at roughly 120 degree offsets. I came to that connection from looking at an AC power diagram rather than from reading fly neuroscience, which is fairly typical of how this project has developed.

---

## The Torus Shape: Why Toroidal

The network is arranged on a toroidal grid, a 32x32 surface where right connects to left and top connects to bottom. This shape was chosen for three specific reasons.

**No boundaries.** Every node has exactly 8 neighbours. A flat grid has edges where waves die or reflect uncontrollably and nodes near edges behave differently. A torus has no edges. Every node participates equally in the wave dynamics.

**Natural wave circulation.** Waves wrap around and return to their source. A sensor injection at timestep T creates a wave that eventually returns to its origin, a form of autoassociative memory from physics rather than from explicit storage.

**Short interconnect paths.** The maximum distance between any two nodes is 16 steps on a 32x32 grid. A signal injected anywhere reaches the entire network within 16 timesteps. Biological neural circuits favour local connectivity for the same reason; long axons are metabolically expensive and slow.

---

## Impedance Boundaries: Resonant Cavities

The sensor regions use inductance values scaled relative to the profile's bulk L value: IMU boundary L x 1.5, sonic boundary L x 1.3. When a wave crosses from one impedance into another, part reflects back, the same physics as light partially reflecting at a glass surface.

The practical effect is that each sensor region acts as a resonant cavity. Waves injected by the sensor bounce within the region rather than escaping immediately. After injection stops, the residual oscillation persists for several cycles, sustained further by the tunnel diode negative resistance.

Making boundaries profile-relative (L x 1.5 rather than an absolute value) ensures every thermodynamic profile experiences the same proportional impedance contrast. Without this, the CHAOTIC profile (L = 0.001) received 40x boundary contrast while NORMAL (L = 0.010) received only 4x, which accidentally amplified CHAOTIC dynamics and inverted the expected profile ordering.

---

## Learning: Hebbian Plasticity and the Mushroom Body

**Neurons that fire together wire together.** When two connected nodes co-fire, the connection between them strengthens. Unused connections slowly weaken. No global error signal. No backward pass.

Two refinements beyond basic Hebbian learning:

**Physarum-inspired superlinear strengthening.** The eligibility trace is raised to a power greater than 1 before the weight update. High-traffic connections grow superlinearly into highways. Rarely used connections atrophy faster. Inspired by the tube conductance rule in Physarum polycephalum slime mould (Tero et al. 2010).

**Reward modulation.** A scalar reward signal modulates the eligibility trace. Good outcomes reinforce recent co-firing patterns. Bad outcomes weaken them. This is the computational equivalent of dopaminergic neuromodulation in the fly mushroom body, which is the brain's mechanism for associating neural activity patterns with outcomes.

---

## Inhibition: The GABA Equivalent

Excitation alone doesn't produce useful computation. Without inhibition, excitatory waves propagate unchecked and the whole network saturates. The brain requires a constant balance between excitation and inhibition.

fly_brain implements two inhibitory mechanisms:

**Divisive normalisation.** Each timestep, if mean activity exceeds TARGET_ACTIVITY (1.5V), all currents are scaled down proportionally. This mirrors cortical gain control.

**Lateral inhibition.** When a node fires it suppresses current in its 8 immediate neighbours. This sharpens spatial contrast and mirrors motion edge detection in the fly optic lobe.

The balance point keeps the network operating where wave dynamics are richest and most discriminable. GABA parameters require calibration. Too strong and the inhibition suppresses the signal itself rather than just the noise. The current parameters (TARGET_ACTIVITY = 1.5, GABA_LATERAL = 0.003) were found to recover all-sensors separation to 0.119 after the initial over-damping reduced it to 0.049.

---

## Memory and Catastrophic Forgetting Prevention

**Experience replay buffer.** A circular buffer stores sparse snapshots of past sensor injection patterns (active nodes only, approximately 20x compression). Every 20 live steps, one past pattern is re-injected and a Hebbian update is performed with reward=0. This is the computational equivalent of hippocampal replay during sleep; the buffer stores the injection pattern that generated a wave state rather than the full torus state itself, which is the same principle as storing a procedural seed rather than the full generated world.

**Elastic Weight Consolidation.** Connections that have been frequently potentiated accumulate importance scores and resist change when the drone moves to a new environment. The motor readout uses a stronger anchoring force (READOUT_EWC_STRENGTH = 0.8) because testing revealed the fast-learning readout (LR = 0.05) is the primary forgetting mechanism, overwriting environment A patterns 50x faster than the Hebbian weights (LR = 0.001).

Catastrophic forgetting reduced from 100% to approximately 23% degradation. Call consolidate_memory() when the drone enters a new operating environment to anchor the current learned behaviour.

---

## Performance

All results are from simulation. Hardware build in progress.

| Metric | Value |
|--------|-------|
| Memory footprint | 140 KB float32 / 62 KB float16 |
| Processing speed | 362 steps/sec on CPU |
| Target rate | 100 Hz (matches IMU) |
| Learning speed | Useful discrimination at 500 steps (5 seconds) |
| Generalisation (novel noise seed) | sep = 0.536 |
| Sensor dropout, all sensors | sep = 0.119 |
| Thermodynamic profiles | NORMAL = 0.109 vs CHAOTIC = 0.047 |
| Wave memory duration | 200 steps = 2000 ms |
| Catastrophic forgetting (with replay + EWC) | 23% degradation |
| Baseline comparison | fly = 0.537 vs hand-coded = 0.700 |
| Platform | Raspberry Pi Zero 2W, CPU only |
| Dependencies | numpy only |

Validation suite: 7/7 pass/fail tests passing. 20/20 unit tests passing.

---

## Comparison to Other Small AI Systems

| Property | Standard CNN | RNN/LSTM | Reservoir/ESN | SNN (LIF) | fly_brain |
|----------|-------------|----------|---------------|-----------|-----------|
| Neuron model | Weighted sum | Weighted sum | Weighted sum | Integrate-and-fire | RLC oscillator |
| Internal oscillation | No | No | No | No | Yes |
| Temporal memory | No | Explicit | Implicit | Membrane | Wave state 2000ms |
| Learning rule | Backprop | Backprop | Readout only | STDP variants | Hebbian local |
| Learns during deployment | No | No | No | Sometimes | Yes |
| Reservoir rewires itself | n/a | n/a | No | No | Yes |
| Wave propagation geometry | No | No | No | No | Yes |
| GPU required | Usually | Usually | No | Specialised | No |
| RAM at drone scale | MB to GB | MB | KB | KB | 140 KB |

No existing system combines continuous online Hebbian learning, physical oscillator dynamics, toroidal wave geometry, 2000 ms wave memory from circuit physics, and sub-£20 deployment simultaneously.

---

## What It Does Not Do

**Cross-geometry generalisation on day one.** The wave state encodes specific geometric regularities of the training environment rather than abstract obstacle physics. Generalisation builds with varied training experience.

**Full retention across arbitrary environment sequences.** Catastrophic forgetting is reduced to approximately 23% degradation. Full retention remains an open problem. Periodic offline consolidation (replay during idle periods rather than interleaved live replay) is the most likely path to further improvement.

**Global positioning.** fly_brain handles local obstacle reaction only. The companion project dronewarp provides GPS/IMU navigation for global positioning. Both are designed to run simultaneously on a Raspberry Pi Zero 2W.

---

## Repository Structure

```
fly_brain/
|-- config.py              (all hyperparameters)
|-- torus.py               (RLC oscillator network with tunnel diode)
|-- sensors.py             (sensor-to-wave injection)
|-- plasticity.py          (Hebbian learning with Physarum rule and EWC)
|-- readout.py             (wave state to motor commands)
|-- controller.py          (main 100 Hz control loop)
|-- demo.py                (standalone simulation demonstration)
|-- requirements.txt       (numpy >= 1.24)
|-- hardware/
|   |-- ultrasonic_driver.py   (HC-SR04 array with simulation fallback)
|   |-- esc_output.py          (ESC PWM driver for GPIO, PCA9685, or simulate)
|   +-- flybrainIO.py          (100 Hz I/O main loop, CSV log, YAML config)
|-- memory/
|   |-- replay_buffer.py       (sparse hippocampal replay buffer)
|   +-- importance_weights.py  (EWC synaptic importance tracker)
|-- tests/
|   +-- test_fly_brain.py      (20 unit tests, all passing)
|-- simulations/
|   |-- sim_environment.py
|   |-- test_01_generalisation.py
|   |-- test_02_catastrophic_forgetting.py
|   |-- test_03_sensor_dropout.py
|   |-- test_04_learning_curve.py
|   |-- test_05_wave_visualisation.py
|   |-- test_06_thermodynamic_profiles.py
|   |-- test_07_baseline_comparison.py
|   |-- test_08_memory_persistence.py
|   |-- test_09_scaling.py
|   +-- run_all.py
+-- results/
    |-- validation_report.txt
    +-- .gitkeep
```

---

## Running It

```bash
# install
pip install numpy

# standalone demonstration: circular flight with ASCII wave display
python -m fly_brain.demo

# unit tests (20 tests, roughly 7 seconds)
python -m pytest fly_brain/tests/test_fly_brain.py -v

# extended validation suite (9 simulations, roughly 10 minutes)
python -m fly_brain.simulations.run_all

# hardware I/O in simulation mode (no Raspberry Pi required)
python -m fly_brain.hardware.flybrainIO --simulate --duration 30 --verbose

# hardware I/O on real hardware (Raspberry Pi + HC-SR04 + ESC)
python -m fly_brain.hardware.flybrainIO --duration 60
```

---

## Dependencies

```
# required
numpy >= 1.24

# optional (Raspberry Pi hardware only)
RPi.GPIO                          # HC-SR04 ultrasonic sensors
adafruit-circuitpython-pca9685    # I2C ESC output board
pyyaml                            # YAML config support
```

No PyTorch. No TensorFlow. No CUDA. No HuggingFace. No LLM.

---

## Author

**Christian Hayes**
Independent aerospace researcher
ceh303@gmail.com
[github.com/ceh303-tech](https://github.com/ceh303-tech)

*fly_brain is a side project built out of curiosity about whether wave physics rather than backpropagation could produce useful navigation behaviour at embedded hardware scale. The honest answer so far is: partially, with interesting failure modes worth understanding.*

---

## Licence

MIT. Free for student and research use.

---

## References

- Hodgkin and Huxley (1952). Quantitative model of the action potential. The RLC neuron model is a circuit approximation of the Hodgkin-Huxley membrane equations.
- Hebb (1949). *The Organisation of Behaviour*. The original statement of the Hebbian learning rule.
- Tero et al. (2010). Physarum polycephalum tube conductance rule. Superlinear Hebbian reinforcement.
- Douglass et al. (1993). Stochastic resonance in mechanoreceptor neurons. Shot noise enhancing sub-threshold signal detection.
- Goto (1954). Tunnel diode negative resistance and the Goto Gate. Sustained oscillation as binary memory.
- Mahowald and Douglas (1991). Silicon neuron. Physical circuit implementation of biological neuron dynamics.
- Jaeger (2001). Echo State Networks. Reservoir computing as the closest family in the existing literature.
- Esser et al. (2016). TrueNorth neuromorphic chip. Spiking neural networks on embedded hardware.
- Carandini and Heeger (2012). Normalisation as a canonical neural computation. Divisive normalisation in sensory cortex.
- Meinhardt and Gierer (1974). Lateral inhibition and pattern formation. Biological basis of contrast sharpening.
- Kirkpatrick et al. (2017). Elastic Weight Consolidation. Overcoming catastrophic forgetting in neural networks.
- McClelland et al. (1995). Complementary learning systems. Hippocampal-neocortical memory consolidation.
- Ito (2020). Drosophila brain atlas. Ocellus, halteres, Johnston's organ, lobula plate tangential cells.
#   f l y _ b r a i n  
 