# core/interfaces.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid circular imports — only used for type hints
    from core.message import Message, PeerMessage
    from core.metrics import StepMetrics, EpisodeSummary


# ---------------------------------------------------------------------------
# IChannel — Strategy
# ---------------------------------------------------------------------------

class IChannel(ABC):
    """
    Strategy interface for the wireless channel model.
    """

    @abstractmethod
    def snr(
        self,
        tx_pos  : tuple[float, float],
        rx_pos  : tuple[float, float],
        tx_power: float,
    ) -> float:
        """
        Compute the signal-to-noise ratio at rx_pos given tx_pos and tx_power.
        Includes all active jammer interference.
        """
        ...

    @abstractmethod
    def deliver(
        self,
        message : "Message",
        tx_pos  : tuple[float, float],
        rx_pos  : tuple[float, float],
        tx_power: float,
    ) -> tuple["Message | None", float]:
        """
        Attempt to deliver a message.

        Returns
        -------
        (message, snr) if delivered (possibly with noise injected by attackers)
        (None, snr)    if dropped due to SNR below threshold
        """
        ...

    @abstractmethod
    def jammer_noise_at(self, pos: tuple[float, float]) -> float:
        """
        Return total jammer noise power at position pos.
        Used to build agent observations.
        """
        ...

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the channel state by dt seconds (jammer moves, etc.)."""
        ...


# ---------------------------------------------------------------------------
# IAttacker — Strategy
# ---------------------------------------------------------------------------

class IAttacker(ABC):
    """
    Strategy interface for all attacker agents.
    """

    @property
    @abstractmethod
    def pos(self) -> tuple[float, float]:
        """Current 2D position of the attacker in the arena."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """False during pause cycles (duty cycle simulation)."""
        ...

    @abstractmethod
    def step(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Move and update attacker state."""
        ...

    @abstractmethod
    def noise_at(self, pos: tuple[float, float]) -> float:
        """
        Return interference noise power at position pos.
        Jammer: function of distance to pos.
        Flooder/Spoofer: return 0.0 (they don't emit broadband noise).
        """
        ...

    @abstractmethod
    def inject(self, message: "Message") -> "Message | None":
        """
        Optionally corrupt or forge a message in transit.

        Returns
        -------
        Message  - original or modified message (jammer returns original,
                   spoofer may return a forged copy)
        None     - message is dropped entirely (extreme jamming)
        """
        ...


# ---------------------------------------------------------------------------
# IProtocol — Strategy
# ---------------------------------------------------------------------------

class IProtocol(ABC):
    """
    Strategy interface for decentralised security protocols.

    The RL policy triggers the protocol (action = 6).
    The protocol itself executes deterministically.
    This separation is the key design choice:
      - RL learns WHEN to trigger
      - Protocol defines HOW to execute

    Concretions: FreqHop, BanVote, IdRevoke (in protocols/)
    Callers:     NetworkEnv (when action=6 is chosen)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def can_trigger(self, obs: np.ndarray) -> bool:
        """
        Return True if the agent observation justifies triggering this protocol.
        Used to select which protocol to run when action=6 is chosen.
        """
        ...

    @abstractmethod
    def execute(
        self,
        agent_id   : int,
        neighbours : list[int],
        network_state: dict,
    ) -> "ProtocolResult":
        """
        Execute the protocol deterministically.

        Parameters
        ----------
        agent_id      : the agent that triggered the protocol
        neighbours    : list of neighbour agent IDs within comm range
        network_state : shared mutable dict from NetworkEnv
                        (agent positions, bans, CRL, queues, etc.)

        Returns
        -------
        ProtocolResult with success flag and optional target agent ID.
        """
        ...


class ProtocolResult:
    """
    Result of a protocol execution.
    """
    __slots__ = ("success", "target_id", "reason")

    def __init__(
        self,
        success  : bool,
        target_id: int | None = None,
        reason   : str        = "",
    ):
        self.success   = success
        self.target_id = target_id
        self.reason    = reason

    def __repr__(self) -> str:
        return (
            f"ProtocolResult(success={self.success}, "
            f"target={self.target_id}, reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# IMetricsObserver — Observer
# ---------------------------------------------------------------------------

class IMetricsObserver(ABC):
    """
    Observer interface for simulation metrics.
    NetworkEnv notifies all registered observers at each step and episode end.

    Concretions: CSVLogger, ConsoleLogger, PlotCollector (in observers/)
    """

    @abstractmethod
    def on_step(self, metrics: "StepMetrics") -> None:
        """Called at the end of every simulation step."""
        ...

    @abstractmethod
    def on_episode_end(self, summary: "EpisodeSummary") -> None:
        """Called when an episode terminates."""
        ...

    def on_training_start(self, config: dict) -> None:
        """Optional hook called before training begins."""
        pass

    def on_training_end(self) -> None:
        """Optional hook called after training completes."""
        pass