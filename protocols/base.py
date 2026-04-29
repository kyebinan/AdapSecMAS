# protocols/base.py
# =============================================================================
# AdapSecMAS — BaseProtocol
# Template Method pattern: defines the skeleton of execute() in the base class.
# Subclasses implement _run() for their specific protocol logic.
#
# SRP: handles trigger checking, timeout management, and result logging.
#      Protocol-specific logic lives in each concrete subclass.
# =============================================================================

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from core.interfaces import IProtocol, ProtocolResult
from core.constants import (
    SNR_THRESHOLD,
    SNR_MAX_NORM,
    FLOOD_RATE_THRESHOLD,
    THREAT_SCORE_ELEVATED,
)


class BaseProtocol(IProtocol):
    """
    Abstract base for all decentralised security protocols.

    Template Method: execute() is defined here.
    It checks preconditions, calls _run(), and handles timeouts.
    Subclasses implement _run() only.

    Key design principle:
      The RL policy triggers the protocol (action = 6).
      The protocol executes deterministically — no learning inside.
    """

    _TIMEOUT_STEPS: int = 10   # steps before a protocol attempt expires

    def __init__(self):
        self._active_rounds : dict[int, int] = {}  # agent_id → steps remaining

    # ------------------------------------------------------------------
    # IProtocol — name property
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str: ...

    # ------------------------------------------------------------------
    # IProtocol — can_trigger
    # ------------------------------------------------------------------

    @abstractmethod
    def can_trigger(self, obs: np.ndarray) -> bool:
        """
        Return True if the observation justifies triggering this protocol.
        obs layout: [snr_norm, jammer_noise, corrupt_rate, flood_rate,
                     spoof_flag, tx_norm, level_norm, agg_comm...]
        """
        ...

    # ------------------------------------------------------------------
    # Template Method — execute()
    # ------------------------------------------------------------------

    def execute(
        self,
        agent_id     : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> ProtocolResult:
        """
        Template Method: check preconditions, run protocol, handle timeout.
        Subclasses must not override execute() — only _run().
        """
        if not neighbours:
            return ProtocolResult(
                success=False,
                reason="no neighbours in range",
            )

        self._active_rounds[agent_id] = self._TIMEOUT_STEPS
        result = self._run(agent_id, neighbours, network_state)
        self._active_rounds.pop(agent_id, None)
        return result

    @abstractmethod
    def _run(
        self,
        agent_id     : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> ProtocolResult:
        """
        Execute the protocol logic.
        network_state is the shared mutable dict from NetworkEnv.
        """
        ...

    # ------------------------------------------------------------------
    # Protected observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snr_norm(obs: np.ndarray) -> float:
        return float(obs[0])

    @staticmethod
    def _flood_rate(obs: np.ndarray) -> float:
        return float(obs[3])

    @staticmethod
    def _spoof_flag(obs: np.ndarray) -> float:
        return float(obs[4])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"