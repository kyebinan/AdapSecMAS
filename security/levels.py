# security/levels.py
# =============================================================================
# AdapSecMAS — SecurityLevel
# Clean Code G25: named constants instead of magic integers.
# IntEnum allows direct comparison with int (level > 1) and indexing.
# =============================================================================

from __future__ import annotations

from enum import IntEnum


class SecurityLevel(IntEnum):
    """
    Four discrete security levels for the Security Level Machine.

    NORMAL   — no threat detected, noop dominant action
    ELEVATED — single weak signal, light mitigation
    HIGH     — multiple signals, active defence
    CRITICAL — all attacks, full protocol response

    Not part of the POSG game model — lives in agent architecture.
    Deterministic function of o_t^i, not learned by RL.
    """
    NORMAL   = 0
    ELEVATED = 1
    HIGH     = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Action masks per level
# cols = [noop, boost×8, boost×16, rate-limit, quarantine, verify, trigger]
# True = action available at this level
# ---------------------------------------------------------------------------
ACTION_MASKS: dict[SecurityLevel, list[bool]] = {
    SecurityLevel.NORMAL:   [True,  False, False, False, False, False, False],
    SecurityLevel.ELEVATED: [True,  True,  False, True,  False, True,  False],
    SecurityLevel.HIGH:     [True,  True,  True,  True,  True,  True,  False],
    SecurityLevel.CRITICAL: [True,  True,  True,  True,  True,  True,  True ],
}