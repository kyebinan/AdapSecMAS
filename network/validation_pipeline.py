# network/validation_pipeline.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque

from core.message import Message
from core.constants import FLOOD_RATE_THRESHOLD, SPOOF_SEQ_DELTA


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Pure data result of the pipeline."""
    valid  : bool
    reason : str = ""
    handler: str = ""

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------

class MessageHandler(ABC):
    """
    Abstract base for Chain of Responsibility handlers.
    Each handler either rejects the message or passes it to the next.
    """

    def __init__(self):
        self._next: MessageHandler | None = None

    def set_next(self, handler: "MessageHandler") -> "MessageHandler":
        """Link next handler. Returns handler for fluent chaining."""
        self._next = handler
        return handler

    def handle(self, message: Message, receiver_id: int) -> ValidationResult:
        result = self._check(message, receiver_id)
        if result is not None:
            return result   # rejected — stop chain

        if self._next is not None:
            return self._next.handle(message, receiver_id)

        return ValidationResult(valid=True)   # end of chain — all passed

    @abstractmethod
    def _check(
        self,
        message     : Message,
        receiver_id : int,
    ) -> ValidationResult | None:
        """Return ValidationResult to reject, None to pass to next handler."""
        ...

    def reset(self) -> None:
        """Reset state - called at episode start."""
        pass


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------

class SeqCounterHandler(MessageHandler):
    """
    Anti-spoofing: rejects messages with unexpected sequence counter.
    A jump > seq_delta signals a forged message.
    SRP: only checks sequence numbers.
    """

    def __init__(self, seq_delta: int = SPOOF_SEQ_DELTA):
        super().__init__()
        self._seq_delta = seq_delta
        self._expected  : dict[tuple[int, int], int] = {}

    def _check(self, message: Message, receiver_id: int) -> ValidationResult | None:
        key      = (message.sender_id, receiver_id)
        expected = self._expected.get(key)
        actual   = message.seq

        if expected is None:
            self._expected[key] = actual + 1
            return None

        if actual >= expected and (actual - expected) <= self._seq_delta:
            self._expected[key] = actual + 1
            return None

        return ValidationResult(
            valid   = False,
            reason  = f"seq anomaly: expected~{expected}, got {actual}",
            handler = "SeqCounterHandler",
        )

    def reset(self) -> None:
        self._expected.clear()


class RateLimitHandler(MessageHandler):
    """
    Anti-flooding: rejects messages from senders exceeding the rate threshold.
    Uses a sliding window of WINDOW_SIZE steps.
    """

    WINDOW_SIZE: int = 10

    def __init__(self, threshold: float = FLOOD_RATE_THRESHOLD):
        super().__init__()
        self._threshold = threshold
        self._windows   : dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.WINDOW_SIZE)
        )

    def _check(self, message: Message, receiver_id: int) -> ValidationResult | None:
        sender = message.sender_id
        self._windows[sender].append(1)
        rate = sum(self._windows[sender])

        if rate > self._threshold:
            return ValidationResult(
                valid   = False,
                reason  = f"rate {rate:.1f} > threshold {self._threshold}",
                handler = "RateLimitHandler",
            )
        return None

    def reset(self) -> None:
        self._windows.clear()


class SignatureHandler(MessageHandler):
    """
    Anti-spoofing: rejects unsigned messages or invalid signatures.
    In simulation: signature = hash(sender_id, seq, private_key).
    """

    def __init__(self, public_keys: dict[int, int]):
        super().__init__()
        self._public_keys = public_keys

    def _check(self, message: Message, receiver_id: int) -> ValidationResult | None:
        if not message.is_signed():
            return ValidationResult(
                valid   = False,
                reason  = "unsigned message",
                handler = "SignatureHandler",
            )

        key = self._public_keys.get(message.sender_id)
        if key is None:
            return ValidationResult(
                valid   = False,
                reason  = f"unknown sender {message.sender_id}",
                handler = "SignatureHandler",
            )

        expected_sig = hash((message.sender_id, message.seq, key))
        if message.signature != expected_sig:
            return ValidationResult(
                valid   = False,
                reason  = "invalid signature",
                handler = "SignatureHandler",
            )

        return None


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_validation_pipeline(
    public_keys : dict[int, int],
    seq_delta   : int   = SPOOF_SEQ_DELTA,
    rate_limit  : float = FLOOD_RATE_THRESHOLD,
) -> MessageHandler:
    """
    Build the default validation chain:
      SeqCounterHandler --> RateLimitHandler --> SignatureHandler
    """
    seq  = SeqCounterHandler(seq_delta)
    rate = RateLimitHandler(rate_limit)
    sig  = SignatureHandler(public_keys)

    seq.set_next(rate).set_next(sig)
    return seq