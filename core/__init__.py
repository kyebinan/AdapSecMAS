# core/__init__.py

from core.interfaces import (
    IChannel,
    IAttacker,
    IProtocol,
    IMetricsObserver,
    ProtocolResult,
)

from core.message import (
    Message,
    MessageType,
    PeerMessage,
)

from core.metrics import (
    StepMetrics,
    EpisodeSummary,
)

from core.constants import (
    # channel
    SNR_THRESHOLD,
    SNR_MAX_NORM,
    TX_POWER_DEFAULT,
    PATH_LOSS_EXP,
    NOISE_FLOOR,
    # security level thresholds
    THREAT_SCORE_ELEVATED,
    THREAT_SCORE_HIGH,
    THREAT_SCORE_CRITICAL,
    HYSTERESIS_STEPS,
    # MARL
    N_AGENTS,
    N_ACTIONS,
    DIM_OBS,
    DIM_COMM,
    DIM_OBS_TOTAL,
    GAMMA,
    # reward
    ACTION_COST_TABLE,
    TX_MULTIPLIERS,
)

__all__ = [
    # interfaces
    "IChannel",
    "IAttacker",
    "IProtocol",
    "IMetricsObserver",
    "ProtocolResult",
    # message
    "Message",
    "MessageType",
    "PeerMessage",
    # metrics
    "StepMetrics",
    "EpisodeSummary",
    # constants
    "SNR_THRESHOLD",
    "SNR_MAX_NORM",
    "TX_POWER_DEFAULT",
    "PATH_LOSS_EXP",
    "NOISE_FLOOR",
    "THREAT_SCORE_ELEVATED",
    "THREAT_SCORE_HIGH",
    "THREAT_SCORE_CRITICAL",
    "HYSTERESIS_STEPS",
    "N_AGENTS",
    "N_ACTIONS",
    "DIM_OBS",
    "DIM_COMM",
    "DIM_OBS_TOTAL",
    "GAMMA",
    "ACTION_COST_TABLE",
    "TX_MULTIPLIERS",
]