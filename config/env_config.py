# config/env_config.py
# =============================================================================
# AdapSecMAS — Environment configuration
# Clean Code G35: keep configurable data at high levels.
# All numeric values that control the simulation live here.
# Frozen dataclasses — immutable after construction.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from core.constants import (
    N_AGENTS,
    DT_HEADLESS,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    TX_POWER_DEFAULT,
    PATH_LOSS_EXP,
    NOISE_FLOOR,
    SNR_THRESHOLD,
    NEAR_THRESHOLD_WINDOW,
    NEAR_THRESHOLD_MAX_FAIL,
    QUEUE_MAX,
    FLOOD_RATE_THRESHOLD,
    SPOOF_SEQ_DELTA,
    BAN_DURATION_STEPS,
    FREQ_HOP_QUORUM,
    BAN_VOTE_QUORUM,
    REVOKE_QUORUM,
    HYSTERESIS_STEPS,
    THREAT_SCORE_ELEVATED,
    THREAT_SCORE_HIGH,
    THREAT_SCORE_CRITICAL,
)


@dataclass(frozen=True)
class ChannelConfig:
    """Physical wireless channel parameters."""
    tx_power_default    : float = TX_POWER_DEFAULT
    path_loss_exp       : float = PATH_LOSS_EXP
    noise_floor         : float = NOISE_FLOOR
    snr_threshold       : float = SNR_THRESHOLD
    near_window         : float = NEAR_THRESHOLD_WINDOW
    near_max_fail_prob  : float = NEAR_THRESHOLD_MAX_FAIL


@dataclass(frozen=True)
class ArenaConfig:
    """2D simulation arena dimensions."""
    width  : float = ARENA_WIDTH
    height : float = ARENA_HEIGHT


@dataclass(frozen=True)
class ProtocolConfig:
    """Decentralised protocol parameters."""
    freq_hop_quorum  : float = FREQ_HOP_QUORUM
    ban_vote_quorum  : float = BAN_VOTE_QUORUM
    revoke_quorum    : float = REVOKE_QUORUM
    ban_duration     : int   = BAN_DURATION_STEPS


@dataclass(frozen=True)
class SLMConfig:
    """Security Level Machine thresholds."""
    threshold_elevated : float = THREAT_SCORE_ELEVATED
    threshold_high     : float = THREAT_SCORE_HIGH
    threshold_critical : float = THREAT_SCORE_CRITICAL
    hysteresis_steps   : int   = HYSTERESIS_STEPS


@dataclass(frozen=True)
class EnvConfig:
    """
    Top-level environment configuration.
    Passed to NetworkEnv at construction.

    frozen=True: config is immutable after creation —
    prevents accidental modification during training.
    """
    n_agents  : int          = N_AGENTS
    dt        : float        = DT_HEADLESS
    seed      : int | None   = None

    channel   : ChannelConfig  = field(default_factory=ChannelConfig)
    arena     : ArenaConfig    = field(default_factory=ArenaConfig)
    protocols : ProtocolConfig = field(default_factory=ProtocolConfig)
    slm       : SLMConfig      = field(default_factory=SLMConfig)

    def summary(self) -> str:
        return (
            f"EnvConfig("
            f"n_agents={self.n_agents}, "
            f"dt={self.dt}, "
            f"arena={self.arena.width}×{self.arena.height}, "
            f"snr_threshold={self.channel.snr_threshold}"
            f")"
        )