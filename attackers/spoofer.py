# attackers/spoofer.py
# =============================================================================
# AdapSecMAS — SpoofAgent
# Implements IAttacker (Strategy pattern).
# Forges the sender_id of a legitimate agent and manipulates seq counters.
#
# SRP: only responsible for identity spoofing.
#      Does not degrade SNR, does not flood queues.
# =============================================================================

from __future__ import annotations

from attackers.base import BaseAttacker
from core.message import Message, MessageType
from core.constants import (
    SPOOF_SEQ_DELTA,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)


class SpoofAgent(BaseAttacker):
    """
    Spoofing attacker — forges the identity of a legitimate agent.

    Intercepts messages in transit and replaces them with forged copies:
      - sender_id replaced with the victim's ID
      - seq counter manipulated (jump > SPOOF_SEQ_DELTA)
      - signature invalidated (wrong hash)
      - payload optionally altered

    Detected by SeqCounterHandler and SignatureHandler.
    """

    def __init__(
        self,
        victim_id   : int,
        initial_pos : tuple[float, float] = (3 * ARENA_WIDTH / 4, ARENA_HEIGHT / 4),
        seq_delta   : int = SPOOF_SEQ_DELTA,
        duty_cycle  : float = 0.8,   # not always active — realistic
        rng_seed    : int | None = None,
    ):
        super().__init__(initial_pos, speed=0.0, duty_cycle=duty_cycle, rng_seed=rng_seed)
        self._victim_id  = victim_id
        self._seq_delta  = seq_delta
        self._forged_seq : dict[int, int] = {}   # last forged seq per target

    # ------------------------------------------------------------------
    # IAttacker interface
    # ------------------------------------------------------------------

    def noise_at(self, pos: tuple[float, float]) -> float:
        """Spoofer does not emit radio noise."""
        return 0.0

    def inject(self, message: Message) -> Message | None:
        """
        Forge the message if the spoofer is active.
        Returns a forged copy with victim's ID and manipulated seq counter.
        The signature will be invalid — correct hash requires the private key.
        """
        if not self._active:
            return message

        # Only forge messages from the victim's neighbours (opportunistic)
        # In simulation: forge any message in transit with probability 0.5
        if self._rng.random() > 0.5:
            return message

        return self._forge(message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_victim(self, victim_id: int) -> None:
        """Change the spoofed identity at runtime."""
        self._victim_id  = victim_id
        self._forged_seq = {}

    @property
    def victim_id(self) -> int:
        return self._victim_id

    # ------------------------------------------------------------------
    # Template Method — movement
    # ------------------------------------------------------------------

    def _move(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Spoofer is stationary — operates remotely."""
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _forge(self, original: Message) -> Message:
        """
        Build a forged message:
          - sender_id  → victim's ID
          - seq        → manipulated (jump > seq_delta)
          - signature  → wrong (private key unknown to spoofer)
          - payload    → optionally altered
        """
        target_id    = original.sender_id
        last_forged  = self._forged_seq.get(target_id, 0)
        forged_seq   = last_forged + self._seq_delta + self._rng.randint(1, 5)
        self._forged_seq[target_id] = forged_seq

        forged_payload = dict(original.payload)
        forged_payload["__forged__"] = True   # marker for forensic analysis

        return Message(
            sender_id = self._victim_id,   # forge victim's identity
            msg_type  = original.msg_type,
            seq       = forged_seq,        # manipulated counter
            payload   = forged_payload,
            timestamp = original.timestamp,
            signature = self._rng.randint(1, 2**31),  # random invalid signature
        )