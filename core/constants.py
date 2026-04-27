# core/constants.py

from __future__ import annotations

# ---------------------------------------------------------------------------
# Channel / SNR model
# ---------------------------------------------------------------------------
SNR_THRESHOLD: float        = 4.0    # minimum SNR for a packet to be received
SNR_MAX_NORM: float         = 20.0   # normalization factor for snr_norm feature
NEAR_THRESHOLD_WINDOW: float= 1.0    # SNR margin where probabilistic loss applies
NEAR_THRESHOLD_MAX_FAIL: float = 0.9 # max failure probability near threshold
PATH_LOSS_EXP: float        = 0.3    # urban path-loss exponent alpha
NOISE_FLOOR: float          = 1.0    # base ambient noise power
TX_POWER_DEFAULT: float     = 100.0  # default transmit power (W)

# ---------------------------------------------------------------------------
# Jammer
# ---------------------------------------------------------------------------
JAMMER_POWER: float   = 130.0  # W — interference power at source
JAMMER_RADIUS: float  = 430.0  # px — effective jamming radius
JAMMER_SPEED: float   = 60.0   # px/s

# ---------------------------------------------------------------------------
# Flooder
# ---------------------------------------------------------------------------
FLOOD_RATE_NORMAL: float    = 5.0   # msgs/step — normal agent rate
FLOOD_RATE_ATTACK: float    = 50.0  # msgs/step — flooder rate
FLOOD_RATE_THRESHOLD: float = 15.0  # detection threshold
QUEUE_MAX: int              = 100   # max queue length before overflow

# ---------------------------------------------------------------------------
# Spoofer
# ---------------------------------------------------------------------------
SPOOF_SEQ_DELTA: int = 5  # sequence counter jump that signals a forgery

# ---------------------------------------------------------------------------
# Security Level Machine
# ---------------------------------------------------------------------------
THREAT_SCORE_ELEVATED: float = 0.3  # score above which level >= ELEVATED
THREAT_SCORE_HIGH: float     = 0.8  # score above which level >= HIGH
THREAT_SCORE_CRITICAL: float = 1.5  # score above which level = CRITICAL
HYSTERESIS_STEPS: int        = 3    # steps below threshold before descending

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
BAN_VOTE_QUORUM: float  = 0.5   # fraction of votes required for ban
BAN_DURATION_STEPS: int = 50    # how long a ban lasts
REVOKE_QUORUM: float    = 0.667 # 2/3 quorum for ID revocation
FREQ_HOP_QUORUM: float  = 0.5   # fraction of neighbours that must confirm jam

# ---------------------------------------------------------------------------
# MARL / MAPPO
# ---------------------------------------------------------------------------
N_AGENTS: int           = 20
N_ACTIONS: int          = 7    # noop, ×8, ×16, rate-limit, quarantine, verify, trigger
DIM_OBS: int            = 7    # own observation vector size
DIM_COMM: int           = 5    # peer message vector size (m_i_t)
DIM_OBS_TOTAL: int      = DIM_OBS + DIM_COMM  # actor input size

GAMMA: float        = 0.99   # discount factor
LAMBDA_GAE: float   = 0.95   # GAE lambda
CLIP_EPS: float     = 0.20   # PPO clip epsilon
LR_ACTOR: float     = 3e-4
LR_CRITIC: float    = 1e-3
UPDATE_EVERY: int   = 512    # steps between PPO updates
PPO_EPOCHS: int     = 4
BATCH_SIZE: int     = 256
HIDDEN_SIZE: int    = 64

ENT_START: float        = 0.10
ENT_END: float          = 0.005
ENT_ANNEAL_STEPS: int   = 50_000

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

# M - mission success
ALPHA_DELIVERY_RATE: float    = 2.0
ALPHA_LINKS_HEALTHY: float    = 0.1
ALPHA_PROTOCOL_OK: float      = 1.0

# C_sec - security exposure
BETA_JAM_LOSS: float     = 1.0
BETA_QUEUE_OVER: float   = 0.8
BETA_SPOOF_ACCEPT: float = 1.5   # highest - identity attack cascades
BETA_MISMATCH: float     = 2.0   # primary escalation signal
BETA_PROTO_FAIL: float   = 0.5

# C_act - action cost table [level][action]
# rows = SecurityLevel (0=NORMAL … 3=CRITICAL)
# cols = action id     (0=noop … 6=trigger)
ACTION_COST_TABLE: list[list[float]] = [
    # noop   ×8     ×16    rate   quar   verify trigger
    [0.00,  0.04,  0.08,  0.10,  0.30,  0.06,  0.20],  # NORMAL
    [0.00,  0.02,  0.04,  0.05,  0.15,  0.03,  0.10],  # ELEVATED
    [0.00,  0.01,  0.02,  0.02,  0.05,  0.01,  0.05],  # HIGH
    [0.20,  0.01,  0.01,  0.01,  0.02,  0.01,  0.02],  # CRITICAL - noop costs!
]

# ---------------------------------------------------------------------------
# TX power multipliers per action
# ---------------------------------------------------------------------------
TX_MULTIPLIERS: dict[int, float] = {
    0: 1.0,   # noop
    1: 8.0,   # boost ×8
    2: 16.0,  # boost ×16
    3: 1.0,   # rate-limit - no power change
    4: 1.0,   # quarantine - no power change
    5: 1.0,   # verify-nonce - no power change
    6: 1.0,   # trigger-protocol - no power change
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
DT_HEADLESS: float  = 1.0    # 1 sim-second per step - fast training
DT_RENDER: float    = 1/60   # real-time rendering
ARENA_WIDTH: float  = 1120.0
ARENA_HEIGHT: float = 700.0