# attackers/flooder.py
# =============================================================================
# AdapSecMAS — FloodAgent
# Implements IAttacker (Strategy pattern).
# Internal agent that injects frames at high rate to saturate queues.
#
# SRP: only responsible for flooding — generates excess messages.
#      Does not degrade SNR (noise_at returns 0), does not forge identity.
# =============================================================================

from __future__ import annotations

from attackers.base import BaseAttacker
from core.message import Message, MessageType
from core.constants import (
    FLOOD_RATE_ATTACK,
    FLOOD_RATE_NORMAL,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)


class FloodAgent(BaseAttacker):
    """
    Flooding attacker — injects messages at an abnormally high rate.

    The flooder is an internal agent (it has a legitimate-looking ID)
    that sends messages far above the normal rate threshold.
    Detected by RateLimitHandler via sliding window.

    Targets either a specific victim agent or broadcasts to all.
    """

    def __init__(
        self,
        agent_id    : int,
        initial_pos : tuple[float, float] = (ARENA_WIDTH / 4, ARENA_HEIGHT / 4),
        flood_rate  : float = FLOOD_RATE_ATTACK,
        victim_id   : int | None = None,     # None = flood all neighbours
        duty_cycle  : float = 1.0,
        rng_seed    : int | None = None,
    ):
        super().__init__(initial_pos, speed=0.0, duty_cycle=duty_cycle, rng_seed=rng_seed)
        self._agent_id   = agent_id
        self._flood_rate = flood_rate
        self._victim_id  = victim_id
        self._seq        = 0
        self._time_acc   = 0.0
        self._pending    : list[Message] = []

    # ------------------------------------------------------------------
    # IAttacker interface
    # ------------------------------------------------------------------

    def noise_at(self, pos: tuple[float, float]) -> float:
        """Flooder does not emit radio noise — only saturates queues."""
        return 0.0

    def inject(self, message: Message) -> Message | None:
        """Flooder does not alter messages in transit — it generates its own."""
        return message

    # ------------------------------------------------------------------
    # Flood message generation
    # ------------------------------------------------------------------

    def generate_flood_messages(self, dt: float, timestamp: float) -> list[Message]:
        """
        Generate flood messages for this step.
        Called by NetworkEnv.step() to inject into the network.

        Returns a list of junk messages to be delivered to victim(s).
        """
        if not self._active:
            return []

        self._time_acc += dt
        step_duration   = 1.0 / max(self._flood_rate, 1e-6)
        messages        = []

        while self._time_acc >= step_duration:
            messages.append(self._make_flood_message(timestamp))
            self._time_acc -= step_duration

        return messages

    def set_victim(self, victim_id: int | None) -> None:
        """Change the flood target at runtime."""
        self._victim_id = victim_id

    @property
    def victim_id(self) -> int | None:
        return self._victim_id

    @property
    def agent_id(self) -> int:
        return self._agent_id

    # ------------------------------------------------------------------
    # Template Method — movement
    # ------------------------------------------------------------------

    def _move(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Flooder is stationary — no movement."""
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _make_flood_message(self, timestamp: float) -> Message:
        """Build a junk flood message with the flooder's legitimate-looking ID."""
        msg = Message(
            sender_id = self._agent_id,
            msg_type  = MessageType.TASK_REQUEST,   # disguised as normal traffic
            seq       = self._seq,
            payload   = {"junk": True},
            timestamp = timestamp,
        )
        self._seq += 1
        return msg