"""
Online Novelty Detector.

For every new speech embedding, determines whether the input
represents something the system has seen before (familiar) or
something genuinely new (novel).

This runs inline with every input — it is NOT a batch process.
It uses the Prototype Memory's nearest-neighbor distance to make
the determination.

Novelty detection is important because:
    1. Novel inputs are prioritized for the replay buffer.
    2. A sustained burst of novelty may trigger early global discovery.
    3. Novelty rate is a key metric for measuring the system's learning progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class NoveltyEvent:
    """Records a detected novelty event."""

    vector: np.ndarray
    distance_to_nearest: float
    nearest_prototype_id: Optional[str]
    timestamp: datetime


class NoveltyDetector:
    """
    Tracks novelty rate and maintains a buffer of recent novel inputs.

    This module does not itself compute distances — it receives the
    distance and novelty flag from the PrototypeMemory.update() call
    and maintains statistics and a buffer for the discovery engine.

    Parameters:
        buffer_capacity: Maximum number of novel vectors to retain.
        burst_window: Number of recent inputs to consider for burst detection.
        burst_threshold: Fraction of recent inputs that must be novel to
                         trigger a "novelty burst" signal.
    """

    def __init__(
        self,
        buffer_capacity: int = 1000,
        burst_window: int = 50,
        burst_threshold: float = 0.5,
    ):
        self.buffer_capacity = buffer_capacity
        self.burst_window = burst_window
        self.burst_threshold = burst_threshold

        self._buffer: List[NoveltyEvent] = []
        self._recent_flags: List[bool] = []  # Sliding window of novelty flags
        self._total_inputs: int = 0
        self._total_novel: int = 0

    def record(
        self,
        vector: np.ndarray,
        is_novel: bool,
        distance: float,
        nearest_prototype_id: Optional[str] = None,
    ) -> Optional[NoveltyEvent]:
        """
        Record the novelty status of a processed input.

        Args:
            vector: The (D,) input embedding.
            is_novel: Whether the prototype memory flagged this as novel.
            distance: Distance to the nearest prototype.
            nearest_prototype_id: ID of the nearest prototype.

        Returns:
            A NoveltyEvent if the input was novel, else None.
        """
        self._total_inputs += 1

        # Maintain sliding window for burst detection
        self._recent_flags.append(is_novel)
        if len(self._recent_flags) > self.burst_window:
            self._recent_flags.pop(0)

        if not is_novel:
            return None

        self._total_novel += 1

        event = NoveltyEvent(
            vector=vector.copy(),
            distance_to_nearest=distance,
            nearest_prototype_id=nearest_prototype_id,
            timestamp=datetime.now(timezone.utc),
        )

        self._buffer.append(event)
        if len(self._buffer) > self.buffer_capacity:
            self._buffer.pop(0)

        return event

    @property
    def novelty_rate(self) -> float:
        """Overall fraction of inputs flagged as novel."""
        if self._total_inputs == 0:
            return 0.0
        return self._total_novel / self._total_inputs

    @property
    def recent_novelty_rate(self) -> float:
        """Fraction of recent inputs that were novel (burst detection window)."""
        if not self._recent_flags:
            return 0.0
        return sum(self._recent_flags) / len(self._recent_flags)

    @property
    def is_burst(self) -> bool:
        """
        True if the system is experiencing a novelty burst.

        A burst means the system is encountering a sustained stream
        of unfamiliar speech, which may indicate a new speaker,
        language, or acoustic environment.
        """
        return (
            len(self._recent_flags) >= self.burst_window
            and self.recent_novelty_rate >= self.burst_threshold
        )

    def get_novel_vectors(self) -> np.ndarray:
        """Return buffered novel vectors as (N, D) matrix."""
        if not self._buffer:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([e.vector for e in self._buffer], axis=0)

    def clear_buffer(self) -> int:
        """Clear the novelty buffer. Returns the number of events cleared."""
        n = len(self._buffer)
        self._buffer.clear()
        return n

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_inputs(self) -> int:
        return self._total_inputs

    @property
    def total_novel(self) -> int:
        return self._total_novel
