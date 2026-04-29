# attackers/jammer.py
# =============================================================================
# AdapSecMAS — JammerAgent
# Implements IAttacker (Strategy pattern).
# Moves in 2D space and emits a continuous noise field J(x,y,t).
# SNR degradation is computed by WirelessChannel.jammer_noise_at().
#
# SRP: only responsible for jamming — position, noise field, movement.
#      Does not forge messages, does not touch queues.
# =============================================================================

from __future__ import annotations

import math
from enum import Enum, auto

from attackers.base import BaseAttacker
from core.message import Message
from core.constants import (
    JAMMER_POWER,
    JAMMER_RADIUS,
    JAMMER_SPEED,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)


class JammerMode(Enum):
    """Movement strategy of the jammer."""
    ROAM      = auto()   # random walk — covers the arena
    HUNT      = auto()   # moves toward the weakest agent (set externally)


class JammerAgent(BaseAttacker):
    """
    Spatial jammer — emits noise field J(x,y,t).

    Noise model:
      J(pos) = power × (1 - d/radius)²   if d < radius
      J(pos) = 0                          otherwise

    Movement modes:
      ROAM: random walk with occasional direction change
      HUNT: moves toward a target position (set by NetworkEnv)
    """

    _DIRECTION_CHANGE_INTERVAL: float = 3.0   # seconds between direction changes in ROAM

    def __init__(
        self,
        initial_pos  : tuple[float, float] = (ARENA_WIDTH / 2, ARENA_HEIGHT / 2),
        power        : float = JAMMER_POWER,
        radius       : float = JAMMER_RADIUS,
        speed        : float = JAMMER_SPEED,
        mode         : JammerMode = JammerMode.ROAM,
        duty_cycle   : float = 1.0,
        rng_seed     : int | None = None,
    ):
        super().__init__(initial_pos, speed, duty_cycle, rng_seed)
        self._power      = power
        self._radius     = radius
        self._mode       = mode
        self._direction  = self._random_direction()
        self._time_since_turn = 0.0
        self._hunt_target: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # IAttacker interface
    # ------------------------------------------------------------------

    def noise_at(self, pos: tuple[float, float]) -> float:
        """
        Return jammer noise power at position pos.
        Quadratic decay from the jammer's current position.
        """
        if not self._active:
            return 0.0

        dx   = pos[0] - self._pos[0]
        dy   = pos[1] - self._pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist >= self._radius:
            return 0.0

        fraction = 1.0 - (dist / self._radius)
        return self._power * fraction ** 2

    def inject(self, message: Message) -> Message | None:
        """Jammer does not forge messages — only degrades SNR via noise_at()."""
        return message

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_hunt_target(self, target_pos: tuple[float, float]) -> None:
        """
        Set a hunt target position.
        Called by NetworkEnv when the jammer switches to HUNT mode.
        """
        self._hunt_target = target_pos
        self._mode        = JammerMode.HUNT

    def set_mode(self, mode: JammerMode) -> None:
        self._mode = mode

    # ------------------------------------------------------------------
    # Template Method — movement
    # ------------------------------------------------------------------

    def _move(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Move according to current mode."""
        if self._mode == JammerMode.ROAM:
            self._roam(dt, arena_w, arena_h)
        elif self._mode == JammerMode.HUNT and self._hunt_target is not None:
            self._hunt(dt, arena_w, arena_h)

    def _roam(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Random walk — change direction every _DIRECTION_CHANGE_INTERVAL seconds."""
        self._time_since_turn += dt
        if self._time_since_turn >= self._DIRECTION_CHANGE_INTERVAL:
            self._direction       = self._random_direction()
            self._time_since_turn = 0.0

        dx, dy = self._direction
        new_x  = self._pos[0] + dx * self._speed * dt
        new_y  = self._pos[1] + dy * self._speed * dt

        # Bounce off arena walls
        if new_x <= 0 or new_x >= arena_w:
            self._direction = (-dx, dy)
        if new_y <= 0 or new_y >= arena_h:
            self._direction = (dx, -dy)

        self._pos = self._clamp_to_arena(new_x, new_y, arena_w, arena_h)

    def _hunt(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Move toward the hunt target at full speed."""
        tx, ty = self._hunt_target
        dx     = tx - self._pos[0]
        dy     = ty - self._pos[1]
        dist   = math.sqrt(dx * dx + dy * dy)

        if dist < 1.0:
            return   # already at target

        ux     = dx / dist
        uy     = dy / dist
        new_x  = self._pos[0] + ux * self._speed * dt
        new_y  = self._pos[1] + uy * self._speed * dt
        self._pos = self._clamp_to_arena(new_x, new_y, arena_w, arena_h)