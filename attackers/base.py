# attackers/base.py
# =============================================================================
# AdapSecMAS — BaseAttacker
# Template Method pattern: defines the skeleton of step() in the base class.
# Subclasses override _move() for their specific movement behaviour.
#
# SRP: handles position, movement, and duty cycle only.
#      Attack-specific logic lives in each concrete subclass.
# =============================================================================

from __future__ import annotations

import math
import random
from abc import abstractmethod

from core.interfaces import IAttacker
from core.message import Message
from core.constants import ARENA_WIDTH, ARENA_HEIGHT


class BaseAttacker(IAttacker):
    """
    Abstract base for all attacker agents.
    Provides shared movement logic and duty cycle management.

    Template Method: step() is defined here and calls _move(),
    which each subclass implements for its own movement strategy.
    """

    def __init__(
        self,
        initial_pos : tuple[float, float],
        speed       : float,
        duty_cycle  : float = 1.0,   # fraction of time active (1.0 = always)
        rng_seed    : int | None = None,
    ):
        self._pos        = initial_pos
        self._speed      = speed
        self._duty_cycle = duty_cycle
        self._active     = True
        self._rng        = random.Random(rng_seed)
        self._time       = 0.0

    # ------------------------------------------------------------------
    # IAttacker properties
    # ------------------------------------------------------------------

    @property
    def pos(self) -> tuple[float, float]:
        return self._pos

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Template Method — step() calls _move()
    # ------------------------------------------------------------------

    def step(self, dt: float, arena_w: float = ARENA_WIDTH, arena_h: float = ARENA_HEIGHT) -> None:
        """
        Advance attacker state by dt seconds.
        Template Method: updates duty cycle, then calls _move().
        Subclasses must not override step() — only _move().
        """
        self._time   += dt
        self._active  = self._rng.random() < self._duty_cycle
        if self._active:
            self._move(dt, arena_w, arena_h)

    @abstractmethod
    def _move(self, dt: float, arena_w: float, arena_h: float) -> None:
        """
        Update self._pos according to the attacker's movement strategy.
        Called only when the attacker is active.
        """
        ...

    # ------------------------------------------------------------------
    # Default inject — subclasses override for actual injection
    # ------------------------------------------------------------------

    def inject(self, message: Message) -> Message | None:
        """
        Default: no injection — jammer only degrades SNR, does not forge.
        Flooder and Spoofer override this.
        """
        return message

    # ------------------------------------------------------------------
    # Protected movement helpers
    # ------------------------------------------------------------------

    def _clamp_to_arena(
        self,
        x: float,
        y: float,
        arena_w: float,
        arena_h: float,
    ) -> tuple[float, float]:
        """Keep position within arena boundaries."""
        return (
            max(0.0, min(x, arena_w)),
            max(0.0, min(y, arena_h)),
        )

    def _random_direction(self) -> tuple[float, float]:
        """Return a random unit vector."""
        angle = self._rng.uniform(0, 2 * math.pi)
        return math.cos(angle), math.sin(angle)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pos=({self._pos[0]:.0f},{self._pos[1]:.0f}), "
            f"active={self._active})"
        )