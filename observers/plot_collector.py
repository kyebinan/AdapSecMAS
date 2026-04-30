# observers/plot_collector.py
# =============================================================================
# AdapSecMAS — PlotCollector
# Observer pattern: accumulates metrics in memory for post-hoc plotting.
# SRP: only collects data — plotting logic is in separate plot functions.
#
# Usage:
#   collector = PlotCollector()
#   env.subscribe(collector)
#   # ... run training ...
#   collector.plot_training()    # saves figures to logs/plots/
# =============================================================================

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from core.interfaces import IMetricsObserver
from core.metrics    import StepMetrics, EpisodeSummary


class PlotCollector(IMetricsObserver):
    """
    Accumulates step and episode metrics for matplotlib plotting.

    Data is kept in memory — intended for post-training analysis,
    not for streaming during training.

    SRP: only accumulates data.
    Plotting functions are standalone — not methods of this class.
    """

    def __init__(self):
        # Per-step series (appended each step)
        self.steps        : list[int]   = []
        self.rewards      : list[float] = []
        self.delivery_rates: list[float]= []
        self.snr_values   : list[float] = []
        self.msgs_lost_jam: list[int]   = []
        self.queue_overflows: list[int] = []
        self.spoof_accepted : list[int] = []

        # Level distribution per step
        self.pct_normal  : list[float] = []
        self.pct_elevated: list[float] = []
        self.pct_high    : list[float] = []
        self.pct_critical: list[float] = []

        # Per-episode series
        self.episode_rewards      : list[float] = []
        self.episode_delivery     : list[float] = []
        self.episode_proto_success: list[float] = []
        self.episode_numbers      : list[int]   = []

    # ------------------------------------------------------------------
    # IMetricsObserver
    # ------------------------------------------------------------------

    def on_step(self, metrics: StepMetrics) -> None:
        total_agents = max(
            metrics.n_agents_normal + metrics.n_agents_elevated
            + metrics.n_agents_high + metrics.n_agents_critical, 1
        )
        self.steps.append(metrics.step)
        self.snr_values.append(metrics.avg_snr)
        self.msgs_lost_jam.append(metrics.n_msgs_lost_to_jam)
        self.queue_overflows.append(metrics.n_queue_overflows)
        self.spoof_accepted.append(metrics.n_spoof_accepted)
        self.delivery_rates.append(
            metrics.n_msgs_delivered / max(metrics.n_msgs_sent, 1)
        )
        self.pct_normal.append(metrics.n_agents_normal   / total_agents)
        self.pct_elevated.append(metrics.n_agents_elevated / total_agents)
        self.pct_high.append(metrics.n_agents_high       / total_agents)
        self.pct_critical.append(metrics.n_agents_critical / total_agents)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self.episode_numbers.append(summary.episode)
        self.episode_rewards.append(summary.total_reward)
        self.episode_delivery.append(summary.mean_delivery_rate)
        self.episode_proto_success.append(summary.protocol_success_rate)

    # ------------------------------------------------------------------
    # Plotting — requires matplotlib
    # ------------------------------------------------------------------

    def plot_training(self, output_dir: str = "logs/plots") -> None:
        """
        Generate and save training plots.
        Requires matplotlib — import is deferred so the module loads
        without it during headless training.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plots")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Episode reward
        self._plot_series(
            x     = self.episode_numbers,
            y     = self.episode_rewards,
            title = "Episode Reward",
            xlabel= "Episode",
            ylabel= "Total reward",
            path  = os.path.join(output_dir, "episode_reward.png"),
        )

        # 2. Delivery rate per episode
        self._plot_series(
            x     = self.episode_numbers,
            y     = self.episode_delivery,
            title = "Mean Delivery Rate per Episode",
            xlabel= "Episode",
            ylabel= "Delivery rate",
            ylim  = (0, 1),
            path  = os.path.join(output_dir, "delivery_rate.png"),
        )

        # 3. Security level distribution (stacked area)
        self._plot_level_distribution(output_dir)

        # 4. Attack impact (msgs lost, overflows, spoof)
        self._plot_attack_impact(output_dir)

        print(f"Plots saved to {output_dir}/")

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_series(
        x     : list,
        y     : list,
        title : str,
        xlabel: str,
        ylabel: str,
        path  : str,
        ylim  : tuple | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, linewidth=0.8, alpha=0.7)

        # Smoothed curve
        if len(y) >= 10:
            window = min(20, len(y) // 5)
            smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            ax.plot(
                x[window - 1:], smooth,
                linewidth=2, color="tab:red", label=f"MA({window})"
            )
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_level_distribution(self, output_dir: str) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        steps = list(range(len(self.pct_normal)))
        ax.stackplot(
            steps,
            self.pct_normal,
            self.pct_elevated,
            self.pct_high,
            self.pct_critical,
            labels  = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"],
            colors  = ["#1D9E75", "#EF9F27", "#E24B4A", "#7B2D8B"],
            alpha   = 0.8,
        )
        ax.set_title("Security Level Distribution over Training")
        ax.set_xlabel("Step")
        ax.set_ylabel("Fraction of agents")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "level_distribution.png"), dpi=150)
        plt.close(fig)

    def _plot_attack_impact(self, output_dir: str) -> None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        steps = list(range(len(self.msgs_lost_jam)))

        data = [
            (axes[0], self.msgs_lost_jam,    "Msgs lost to Jamming",   "tab:red"),
            (axes[1], self.queue_overflows,   "Queue Overflows (Flood)","tab:orange"),
            (axes[2], self.spoof_accepted,    "Spoofed msgs Accepted",  "tab:purple"),
        ]
        for ax, series, title, color in data:
            ax.plot(steps, series, linewidth=0.6, alpha=0.5, color=color)
            if len(series) >= 10:
                w = min(20, len(series) // 5)
                sm = np.convolve(series, np.ones(w) / w, mode="valid")
                ax.plot(
                    steps[w - 1:], sm,
                    linewidth=2, color=color, label=f"MA({w})"
                )
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "attack_impact.png"), dpi=150)
        plt.close(fig)