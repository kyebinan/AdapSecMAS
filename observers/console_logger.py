# observers/console_logger.py
# =============================================================================
# AdapSecMAS — ConsoleLogger
# Observer pattern: prints human-readable metrics to stdout.
# SRP: only responsible for console output — no file I/O, no computation.
# =============================================================================

from __future__ import annotations

from core.interfaces import IMetricsObserver
from core.metrics    import StepMetrics, EpisodeSummary


class ConsoleLogger(IMetricsObserver):
    """
    Prints a one-line summary every `log_every` steps
    and a full episode summary at episode end.

    SRP: only prints — no file I/O, no aggregation.
    """

    def __init__(self, log_every: int = 100):
        self._log_every  = log_every
        self._step_count = 0

    # ------------------------------------------------------------------
    # IMetricsObserver
    # ------------------------------------------------------------------

    def on_step(self, metrics: StepMetrics) -> None:
        self._step_count += 1
        if self._step_count % self._log_every != 0:
            return

        delivery = (
            metrics.n_msgs_delivered / max(metrics.n_msgs_sent, 1)
        )
        print(
            f"  step {metrics.step:>5}  "
            f"snr={metrics.avg_snr:>6.2f}  "
            f"delivery={delivery:.1%}  "
            f"jam={metrics.n_msgs_lost_to_jam:>3}  "
            f"flood={metrics.n_queue_overflows:>3}  "
            f"spoof={metrics.n_spoof_accepted:>3}  "
            f"proto_ok={metrics.n_protocol_success:>3}  "
            f"[N={metrics.n_agents_normal} "
            f"E={metrics.n_agents_elevated} "
            f"H={metrics.n_agents_high} "
            f"C={metrics.n_agents_critical}]"
        )

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        bar = self._make_level_bar(
            summary.pct_normal,
            summary.pct_elevated,
            summary.pct_high,
            summary.pct_critical,
        )
        print(
            f"\n── Episode {summary.episode} ──────────────────────────\n"
            f"  reward        : {summary.total_reward:>10.2f}\n"
            f"  delivery rate : {summary.mean_delivery_rate:>10.2%}  "
            f"(min {summary.min_delivery_rate:.2%})\n"
            f"  msgs lost/jam : {summary.total_msgs_lost_to_jam:>10}\n"
            f"  queue overflow: {summary.total_queue_overflows:>10}\n"
            f"  spoof accepted: {summary.total_spoof_accepted:>10}\n"
            f"  spoof blocked : {summary.total_spoof_blocked:>10}\n"
            f"  proto success : {summary.protocol_success_rate:>10.2%}\n"
            f"  level dist    : {bar}\n"
        )

    def on_training_start(self, config: dict) -> None:
        print("\n" + "=" * 60)
        print("  AdapSecMAS — MAPPO Training")
        for k, v in config.items():
            print(f"  {k:<20}: {v}")
        print("=" * 60 + "\n")

    def on_training_end(self) -> None:
        print("\n" + "=" * 60)
        print("  Training complete.")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _make_level_bar(
        pct_n: float,
        pct_e: float,
        pct_h: float,
        pct_c: float,
        width: int = 30,
    ) -> str:
        """
        Visual bar showing time distribution across security levels.
        Example: [NNNNNEEEEHHHCC        ]
        """
        n = max(1, int(pct_n * width))
        e = max(0, int(pct_e * width))
        h = max(0, int(pct_h * width))
        c = max(0, int(pct_c * width))
        bar = "N" * n + "E" * e + "H" * h + "C" * c
        bar = bar[:width].ljust(width)
        return f"[{bar}] N={pct_n:.0%} E={pct_e:.0%} H={pct_h:.0%} C={pct_c:.0%}"