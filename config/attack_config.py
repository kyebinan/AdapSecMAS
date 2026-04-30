# config/attack_config.py
# =============================================================================
# AdapSecMAS — Attack configuration
# Parameters for each attacker family.
# frozen=True: immutable — attackers read config at construction only.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from core.constants import (
    JAMMER_POWER,
    JAMMER_RADIUS,
    JAMMER_SPEED,
    FLOOD_RATE_ATTACK,
    FLOOD_RATE_NORMAL,
    SPOOF_SEQ_DELTA,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)


@dataclass(frozen=True)
class JammerConfig:
    """Spatial jammer parameters."""
    power        : float                    = JAMMER_POWER
    radius       : float                    = JAMMER_RADIUS
    speed        : float                    = JAMMER_SPEED
    initial_pos  : tuple[float, float]      = field(
        default_factory=lambda: (ARENA_WIDTH / 2, ARENA_HEIGHT / 2)
    )
    duty_cycle   : float = 1.0   # fraction of steps the jammer is active


@dataclass(frozen=True)
class FlooderConfig:
    """MAC flooding attacker parameters."""
    agent_id    : int                  = 20   # ID outside protagonist range
    flood_rate  : float                = FLOOD_RATE_ATTACK
    normal_rate : float                = FLOOD_RATE_NORMAL
    victim_id   : int | None           = None  # None = flood all
    initial_pos : tuple[float, float]  = field(
        default_factory=lambda: (ARENA_WIDTH / 4, ARENA_HEIGHT / 4)
    )
    duty_cycle  : float = 1.0


@dataclass(frozen=True)
class SpooferConfig:
    """Identity spoofing attacker parameters."""
    victim_id   : int                  = 0    # agent whose ID is forged
    seq_delta   : int                  = SPOOF_SEQ_DELTA
    initial_pos : tuple[float, float]  = field(
        default_factory=lambda: (3 * ARENA_WIDTH / 4, ARENA_HEIGHT / 4)
    )
    duty_cycle  : float = 0.8   # not always active — realistic


@dataclass(frozen=True)
class AttackConfig:
    """
    Top-level attack configuration.
    Controls which attackers are active and their parameters.

    active_* flags are set by the curriculum in train.py.
    At construction all are True — curriculum overrides at runtime.
    """
    jammer_active : bool = True
    flooder_active: bool = True
    spoofer_active: bool = True

    jammer  : JammerConfig   = field(default_factory=JammerConfig)
    flooder : FlooderConfig  = field(default_factory=FlooderConfig)
    spoofer : SpooferConfig  = field(default_factory=SpooferConfig)

    def summary(self) -> str:
        active = []
        if self.jammer_active:  active.append("jammer")
        if self.flooder_active: active.append("flooder")
        if self.spoofer_active: active.append("spoofer")
        return f"AttackConfig(active={active or 'none'})"