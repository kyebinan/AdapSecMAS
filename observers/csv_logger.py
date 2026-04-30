# observers/csv_logger.py
# =============================================================================
# AdapSecMAS — CSVLogger
# Observer pattern: logs StepMetrics and EpisodeSummary to CSV files.
# SRP: only responsible for CSV writing — no computation, no display.
# =============================================================================

from __future__ import annotations

import csv
import os
from dataclasses import asdict

from core.interfaces import IMetricsObserver
from core.metrics    import StepMetrics, EpisodeSummary


class CSVLogger(IMetricsObserver):
    """
    Writes step and episode metrics to two separate CSV files.

    Files are opened once at construction and closed on flush() or __del__.
    Headers are written automatically from the dataclass field names.

    SRP: only writes to disk — no formatting, no aggregation.
    """

    def __init__(
        self,
        step_path   : str = "logs/train_steps.csv",
        episode_path: str = "logs/train_episodes.csv",
    ):
        os.makedirs(os.path.dirname(step_path),    exist_ok=True)
        os.makedirs(os.path.dirname(episode_path), exist_ok=True)

        self._step_fh    = open(step_path,    "w", newline="", buffering=1)
        self._episode_fh = open(episode_path, "w", newline="", buffering=1)

        # Writers initialised lazily on first write (field names from dataclass)
        self._step_writer   : csv.DictWriter | None = None
        self._episode_writer: csv.DictWriter | None = None

    # ------------------------------------------------------------------
    # IMetricsObserver
    # ------------------------------------------------------------------

    def on_step(self, metrics: StepMetrics) -> None:
        row = asdict(metrics)
        if self._step_writer is None:
            self._step_writer = csv.DictWriter(
                self._step_fh, fieldnames=list(row.keys())
            )
            self._step_writer.writeheader()
        self._step_writer.writerow(row)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        row = asdict(summary)
        if self._episode_writer is None:
            self._episode_writer = csv.DictWriter(
                self._episode_fh, fieldnames=list(row.keys())
            )
            self._episode_writer.writeheader()
        self._episode_writer.writerow(row)

    def on_training_end(self) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush and close both CSV files."""
        self._step_fh.flush()
        self._episode_fh.flush()

    def __del__(self) -> None:
        try:
            self._step_fh.close()
            self._episode_fh.close()
        except Exception:
            pass