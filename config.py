# fly_brain/config.py
import math as _math
"""
All hyperparameters for the fly_brain wave navigation controller.

Designed to run on Raspberry Pi Zero 2W (512 MB RAM, CPU only).
Inherited from TTCE NORMAL thermodynamic profile, scaled down to fit embedded target.
"""

# ---------------------------------------------------------------------------
# Torus geometry — MUST fit in Pi Zero 2W (512 MB RAM)
# ---------------------------------------------------------------------------
TORUS_W: int = 32          # Grid width
TORUS_H: int = 32          # Grid height
N_NODES: int = TORUS_W * TORUS_H  # 1,024 nodes

# Three-phase architecture — GPS / IMU / SONIC each on a dedicated torus
# 120° phase separation gives independent wave codes for each sensor modality
N_PHASES: int = 3
PHASE_OFFSETS: list = [0.0,
                       2.0 * _math.pi / 3.0,
                       4.0 * _math.pi / 3.0]   # 0°, 120°, 240°

# Balanced ternary readout trit mapping (most information-dense base, ~ Euler's e)
#   GPS  phase → trit +1  (position / safety)
#   IMU  phase → trit  0  (neutral motion reference)
#   SONIC phase → trit -1  (obstacle / danger)
# trit[i] = (V_gps[i] - V_sonic[i]) / (|V_gps|+|V_imu|+|V_sonic|+ε) ∈ [-1,+1]
TRIT_GPS_POLE:   int = +1
TRIT_IMU_POLE:   int =  0
TRIT_SONIC_POLE: int = -1

# ---------------------------------------------------------------------------
# Three-phase inter-torus coupling
# ---------------------------------------------------------------------------
# Quantum-amplitude cross-phase feedback: fraction of psi_mag injected back
CROSS_PHASE_COUPLING: float = 0.05

# Stochastic resonance cross-injection: GPS noise → IMU/SONIC tori
# Helps each torus detect sub-threshold signals using sibling-phase noise.
SR_NOISE_AMP: float = 0.002

# Leaky-integration decay for psi (quantum interference working memory)
COHERENCE_DECAY: float = 0.95

# Physarum slime-mold superlinear Hebbian reinforcement
# power 1.0 = standard Hebbian; >1 → high-traffic connections grow into highways
SLIME_MOLD_POWER: float = 1.3

# Sub-neurons per node (reduced from TTCE's 16)
N_SUB: int = 4
N_TOTAL: int = N_NODES * N_SUB  # 4,096 total oscillators

# ---------------------------------------------------------------------------
# RLC dynamics — NORMAL profile inherited from TTCE
# ---------------------------------------------------------------------------
R: float = 0.10            # Resistance — damping
L: float = 0.01            # Inductance — sets oscillation frequency
C: float = 0.01            # Capacitance — charge storage

# Negative-resistance oscillator core
V_CRIT: float = 2.0        # Centre of the negative-resistance region
ALPHA:  float = 0.05       # Energy-pump strength within the active region

# Stochastic Goto Gate — kept exactly from TTCE
BETA:    float = 5.0       # Fermi-Dirac steepness
V_TUNNEL: float = 1.5      # Tunneling threshold voltage
THERMAL_NOISE: float = 0.01  # Thermal noise std dev
SHOT_NOISE_AMP: float = 0.002  # Shot-noise amplitude on current updates
GABA_LATERAL: float = 0.003  # Lateral inhibition gain for firing-node neighbour suppression
TARGET_ACTIVITY: float = 1.5  # Divisive-normalization target for mean abs voltage

# Timestep — 100 Hz to match typical IMU rate
DT: float = 0.01

# Torus feedback — fraction of routed neighbour signal fed back
# Keep small to ensure dissipation; waves propagate but don't self-sustain
TORUS_FEEDBACK: float = 0.005

# ---------------------------------------------------------------------------
# Sparse connectivity
# Memory: 1024 × 8 × fp16 = 16 KB
# ---------------------------------------------------------------------------
CONNECTIONS_PER_NODE: int = 8   # 8-connected local neighbourhood (N/S/E/W + diagonals)

# ---------------------------------------------------------------------------
# Hebbian learning
# ---------------------------------------------------------------------------
HEBB_LR: float = 0.001         # Synaptic strengthening rate
HEBB_DECAY: float = 0.0001     # Synaptic decay per step (forgetting)
ELIGIBILITY_DECAY: float = 0.95  # Eligibility trace decay per step
MAX_WEIGHT: float = 5.0        # Maximum connection weight (clamp)

# ---------------------------------------------------------------------------
# Motor readout
# ---------------------------------------------------------------------------
N_MOTORS: int = 4              # Quadrotor — 4 rotors
READOUT_LR: float = 0.05       # Readout weight learning rate

# ---------------------------------------------------------------------------
# Sensor injection regions
# The 32x32 torus is divided into quadrants, one per sensor type.
# (x0, y0, x1, y1) — pixel coordinates, exclusive upper bound.
# ---------------------------------------------------------------------------
GPS_REGION   = (0,  0,  16, 16)   # Top-left:     GPS position
IMU_REGION   = (16, 0,  32, 16)   # Top-right:    IMU acceleration/gyro
SONIC_REGION = (0,  16, 16, 32)   # Bottom-left:  Ultrasonic ranging
FLOW_REGION  = (16, 16, 32, 32)   # Bottom-right: Optical flow (reserved)

# ---------------------------------------------------------------------------
# Thermodynamic profiles — used by FlyBrainTorus(profile='...') and Test 6
# ---------------------------------------------------------------------------
THERMODYNAMIC_PROFILES = {
    'FROZEN':  {'R': 0.30, 'L': 0.08, 'C': 0.08, 'V_CRIT': 5.0,
                'ALPHA': 0.02, 'BETA': 2.0, 'V_TUNNEL': 3.0},
    'CALM':    {'R': 0.15, 'L': 0.03, 'C': 0.03, 'V_CRIT': 3.0,
                'ALPHA': 0.03, 'BETA': 3.0, 'V_TUNNEL': 2.0},
    'NORMAL':  {'R': 0.10, 'L': 0.01, 'C': 0.01, 'V_CRIT': 2.0,
                'ALPHA': 0.05, 'BETA': 5.0, 'V_TUNNEL': 1.5},
    'EXCITED': {'R': 0.05, 'L': 0.004,'C': 0.004,'V_CRIT': 1.2,
                'ALPHA': 0.08, 'BETA': 7.0, 'V_TUNNEL': 0.9},
    'CHAOTIC': {'R': 0.02, 'L': 0.001,'C': 0.001,'V_CRIT': 0.7,
                'ALPHA': 0.12, 'BETA': 9.0, 'V_TUNNEL': 0.5},
}

PROFILES = {
    "FROZEN":  dict(R=0.50, L=0.05, C=0.05, V_CRIT=5.0, ALPHA=0.01, BETA=1.0),
    "CALM":    dict(R=0.20, L=0.02, C=0.02, V_CRIT=3.0, ALPHA=0.02, BETA=2.0),
    "NORMAL":  dict(R=0.10, L=0.01, C=0.01, V_CRIT=2.0, ALPHA=0.05, BETA=5.0),
    "EXCITED": dict(R=0.05, L=0.005,C=0.005,V_CRIT=1.5, ALPHA=0.10, BETA=8.0),
    "CHAOTIC": dict(R=0.02, L=0.002,C=0.002,V_CRIT=1.0, ALPHA=0.20, BETA=12.0),
}

# ---------------------------------------------------------------------------
# Reward and safety thresholds
# ---------------------------------------------------------------------------
SAFE_RANGE: float = 1.5    # metres — minimum obstacle distance for reward +1

# ---------------------------------------------------------------------------
# Hardware I/O (fly_brain/hardware/)
# ---------------------------------------------------------------------------
N_ULTRASONIC_SENSORS: int  = 8       # Maximum HC-SR04 sensors in the array
SAFE_RANGE_M:        float = 1.5     # metres — alias for SAFE_RANGE used by hardware layer
ESC_MIN_PULSE_US:    int   = 1000    # microseconds — minimum ESC pulse (armed / minimum throttle)
ESC_MAX_PULSE_US:    int   = 2000    # microseconds — maximum ESC pulse (full throttle)
PWM_FREQUENCY_HZ:    int   = 400     # PWM carrier frequency in Hz (standard for digital ESCs)

# ──────────────────────────────────────────────────────────────────
# Experience Replay — catastrophic forgetting prevention
# ──────────────────────────────────────────────────────────────────

# Replay buffer capacity — number of past snapshots to retain
# Memory cost: capacity × mean_active_nodes × 4 bytes
# At 5% activity (51 nodes): 500 × 51 × 4 = ~102 KB
REPLAY_BUFFER_SIZE: int = 500

# Replay one past pattern every N live steps
# Lower = more replay, better retention, more compute
REPLAY_EVERY_N_STEPS: int = 20

# Minimum buffer size before replay begins
REPLAY_MIN_BUFFER: int = 50

# Only store injection states where at least one node exceeds this
# amplitude. Prevents storing near-zero states that add no signal.
# Set relative to V_CRIT to be profile-independent.
REPLAY_STORE_THRESHOLD: float = 0.1  # volts

# ──────────────────────────────────────────────────────────────────
# Elastic Weight Consolidation — secondary forgetting prevention
# ──────────────────────────────────────────────────────────────────

# How fast importance scores accumulate from Hebbian activity
EWC_IMPORTANCE_LR: float = 0.01

# How fast importance scores decay when connections are unused
EWC_IMPORTANCE_DECAY: float = 0.0001

# How strongly important weights resist change
# 0.0 = no EWC effect, 1.0 = important weights fully frozen
# Start low (0.05) and increase if forgetting persists
EWC_CONSOLIDATION_STRENGTH: float = 0.05

# EWC strength specifically for the readout (delta-rule) layer.
# Higher than Hebbian EWC because READOUT_LR (0.05) >> HEBB_LR (0.001).
# The readout needs proportionally stronger protection to resist the same
# number of overwrite steps. Only active after consolidate_memory() is called.
READOUT_EWC_STRENGTH: float = 0.8
