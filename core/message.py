# core/message.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MessageType(Enum):
    """Enumerates all message types in the network."""
    # -- Delivery coordination (functional layer) --
    TASK_REQUEST  = auto()
    TASK_ASSIGN   = auto()
    TASK_ARRIVED  = auto()

    # -- Ride workflow (functional layer) --
    REQUEST_RIDE       = auto()
    OFFER_RIDE         = auto()
    ACCEPT_RIDE        = auto()
    RIDE_DROPOFF       = auto()
    DRONE_READY_PICKUP = auto()
    RIDE_RETURNED      = auto()

    # -- Security protocols (security layer) --
    HOP_REQUEST  = auto()   # FREQ-HOP: propose channel switch
    HOP_ACK      = auto()   # FREQ-HOP: confirm SNR also degraded
    BAN_PROPOSAL = auto()   # BAN-VOTE: nominate suspect
    BAN_VOTE     = auto()   # BAN-VOTE: cast signed vote
    REVOKE_REQ   = auto()   # ID-REVOKE: request identity revocation
    REVOKE_ACK   = auto()   # ID-REVOKE: confirm proof locally

    # -- Peer communication (MARL comm layer) --
    PEER_STATE   = auto()   # m_i_t: broadcast own network state to neighbours


@dataclass(frozen=True)
class Message:
    """
    Attributes
    ----------
    sender_id   : unique identifier of the sending agent
    msg_type    : semantic type of the message
    seq         : sequence counter - incremented per sender per message
                  used by SeqCounterHandler to detect spoofing
    payload     : type-specific content (dict for flexibility)
    timestamp   : simulation time at emission (seconds)
    signature   : simplified hash representing a cryptographic signature
                  In production: ECDSA over (sender_id, seq, payload)
                  In simulation: hash(sender_id, seq) for lightweight spoofing detection
    """
    sender_id : int
    msg_type  : MessageType
    seq       : int
    payload   : dict[str, Any]
    timestamp : float
    signature : int = field(default=0)  # 0 = not yet signed

    def sign(self, private_key: int) -> "Message":
        """
        Return a new Message with the signature field set.
        Uses a lightweight hash - replace with ECDSA for production.
        """
        sig = hash((self.sender_id, self.seq, private_key))
        return Message(
            sender_id = self.sender_id,
            msg_type  = self.msg_type,
            seq       = self.seq,
            payload   = self.payload,
            timestamp = self.timestamp,
            signature = sig,
        )

    def is_signed(self) -> bool:
        return self.signature != 0

    def __repr__(self) -> str:
        return (
            f"Message(from={self.sender_id}, "
            f"type={self.msg_type.name}, "
            f"seq={self.seq})"
        )

@dataclass(frozen=True)
class PeerMessage:
    """
    RL communication vector m_i_t broadcast to neighbours.

    This is NOT a network message in the protocol sense.
    It carries the local network state of agent i for MARL coordination.
    It travels through GossipMediator and is degraded by jammer noise.

    """
    sender_id     : int
    snr_norm      : float   # own SNR / SNR_MAX_NORM
    corrupt_rate  : float   # PER over last 5 messages
    flood_rate    : float   # incoming msg/step from suspects
    spoof_flag    : float   # 1.0 if seq anomaly detected, else 0.0
    level_norm    : float   # SecurityLevel / 3.0

    def to_array(self) -> list[float]:
        """Return as ordered list for obs aggregation."""
        return [
            self.snr_norm,
            self.corrupt_rate,
            self.flood_rate,
            self.spoof_flag,
            self.level_norm,
        ]

    def __repr__(self) -> str:
        return (
            f"PeerMsg(from={self.sender_id}, "
            f"snr={self.snr_norm:.2f}, "
            f"corrupt={self.corrupt_rate:.2f}, "
            f"flood={self.flood_rate:.2f}, "
            f"spoof={self.spoof_flag:.0f}, "
            f"level={self.level_norm:.2f})"
        )