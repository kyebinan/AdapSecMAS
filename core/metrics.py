# core/metrics.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class StepMetrics:
    """
    Snapshot of one simulation step - produced by NetworkEnv.step().

    All counters are team-level (sum over all agents this step).
    """

    # -- step info --
    step       : int   = 0
    sim_time   : float = 0.0

    # -- channel quality --
    avg_snr          : float = 0.0   # mean SNR over all active links
    n_links_healthy  : int   = 0     # links with SNR > SNR_THRESHOLD
    n_links_total    : int   = 0

    # -- jamming --
    n_msgs_sent          : int   = 0
    n_msgs_delivered     : int   = 0
    n_msgs_lost_to_jam   : int   = 0  # SNR below threshold --> dropped
    n_msgs_corrupted     : int   = 0  # near-threshold probabilistic loss

    # -- flooding --
    n_queue_overflows    : int   = 0  # agents whose queue hit QUEUE_MAX
    avg_queue_pressure   : float = 0.0

    # -- spoofing --
    n_spoof_attempts     : int   = 0  # messages with forged sender_id
    n_spoof_accepted     : int   = 0  # spoofed messages that passed validation
    n_spoof_blocked      : int   = 0  # spoofed messages caught by SeqHandler

    # -- protocols --
    n_protocol_triggered : int   = 0  # agents that triggered a protocol
    n_protocol_success   : int   = 0  # protocols that reached quorum
    n_protocol_failed    : int   = 0  # protocols that timed out / no quorum

    # -- security level --
    n_agents_normal   : int = 0
    n_agents_elevated : int = 0
    n_agents_high     : int = 0
    n_agents_critical : int = 0
    level_mismatch    : float = 0.0   # mean gap between level and threat

    # -- MARL actions (this step) --
    n_noop       : int = 0
    n_boost_x8   : int = 0
    n_boost_x16  : int = 0
    n_rate_limit : int = 0
    n_quarantine : int = 0
    n_verify     : int = 0
    n_trigger    : int = 0

    # -- delta metrics (vs previous step) --
    delta_delivery_rate   : float = 0.0   # change in fraction delivered
    delta_protocol_success: float = 0.0   # change in protocol success rate

    def delivery_rate(self) -> float:
        """Fraction of sent messages that were delivered."""
        if self.n_msgs_sent == 0:
            return 1.0
        return self.n_msgs_delivered / self.n_msgs_sent

    def to_dict(self) -> dict:
        """Serialise for CSV logging."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"StepMetrics(step={self.step}, "
            f"snr={self.avg_snr:.2f}, "
            f"delivered={self.n_msgs_delivered}/{self.n_msgs_sent}, "
            f"jam={self.n_msgs_lost_to_jam}, "
            f"flood={self.n_queue_overflows}, "
            f"spoof={self.n_spoof_accepted})"
        )


@dataclass
class EpisodeSummary:
    """
    Aggregated summary over one full episode.
    Produced at episode end by NetworkEnv.

    """

    episode     : int   = 0
    total_steps : int   = 0
    total_reward: float = 0.0

    # delivery
    mean_delivery_rate    : float = 0.0
    min_delivery_rate     : float = 1.0
    total_msgs_sent       : int   = 0
    total_msgs_delivered  : int   = 0

    # jamming impact
    total_msgs_lost_to_jam: int   = 0
    mean_snr              : float = 0.0

    # flooding impact
    total_queue_overflows : int   = 0

    # spoofing impact
    total_spoof_accepted  : int   = 0
    total_spoof_blocked   : int   = 0

    # protocol effectiveness
    total_protocols_triggered: int   = 0
    total_protocols_success  : int   = 0
    protocol_success_rate    : float = 0.0

    # security level distribution (% of steps in each level)
    pct_normal   : float = 0.0
    pct_elevated : float = 0.0
    pct_high     : float = 0.0
    pct_critical : float = 0.0

    # detection quality
    false_positive_rate : float = 0.0   # protocols triggered with no attack
    mean_level_mismatch : float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)