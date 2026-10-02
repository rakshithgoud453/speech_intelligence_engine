"""
Replay Buffer — Reservoir Sampling.

Maintains a fixed-size buffer of past speech experiences for use
in continual learning. The buffer is updated using reservoir sampling,
which guarantees that at any point in time, every experience the system
has ever seen has an equal probability of being in the buffer.

Algorithm Reference:
    Vitter, J.S. (1985). "Random Sampling with a Reservoir."
    ACM Transactions on Mathematical Software.

This is the standard replay strategy for continual learning
(used in Experience Replay, Dark Experience Replay, etc.).

The buffer stores pooled embedding vectors paired with metadata
needed to reconstruct the training signal (source audio references).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ReplayEntry:
    """
    A single entry in the replay buffer.

    Stores enough information to reconstruct the training signal:
    the embedding vector and a reference to the source audio.
    """

    vector: np.ndarray              # (D,) pooled embedding
    model_version: str              # Which encoder produced this
    source_recording_id: Optional[str]
    source_start_ms: int
    source_end_ms: int
    created_at: datetime
    is_novel: bool                  # Was this flagged as novel when first seen?


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer using reservoir sampling.

    Guarantees: At any point after seeing N experiences, each of the
    N experiences has exactly (capacity / N) probability of being
    in the buffer. This is mathematically optimal for maintaining
    a representative sample of an unbounded stream.

    Parameters:
        capacity: Maximum number of entries to retain.
        novelty_reserve_ratio: Fraction of the buffer reserved for novel
                               experiences (0.0 to 1.0). Novel experiences
                               are sampled separately to ensure rare patterns
                               are not lost. Default 0.2 (20%).
    """

    def __init__(
        self,
        capacity: int = 5000,
        novelty_reserve_ratio: float = 0.2,
    ):
        if capacity < 10:
            raise ValueError("Buffer capacity must be at least 10.")
        if not 0.0 <= novelty_reserve_ratio < 1.0:
            raise ValueError("novelty_reserve_ratio must be in [0.0, 1.0).")

        self.capacity = capacity
        self.novelty_reserve_ratio = novelty_reserve_ratio

        # Split into two pools: common and novel
        self._novel_capacity = max(1, int(capacity * novelty_reserve_ratio))
        self._common_capacity = capacity - self._novel_capacity

        self._common_buffer: List[ReplayEntry] = []
        self._novel_buffer: List[ReplayEntry] = []

        self._common_seen: int = 0
        self._novel_seen: int = 0

    def add(
        self,
        vector: np.ndarray,
        model_version: str,
        is_novel: bool = False,
        source_recording_id: Optional[str] = None,
        source_start_ms: int = 0,
        source_end_ms: int = 0,
    ) -> None:
        """
        Add an experience to the replay buffer using reservoir sampling.

        Args:
            vector: The (D,) pooled embedding vector.
            model_version: Encoder version string.
            is_novel: Whether this experience was flagged as novel.
            source_recording_id: Optional source recording reference.
            source_start_ms: Start offset in source recording.
            source_end_ms: End offset in source recording.
        """
        entry = ReplayEntry(
            vector=vector.copy(),
            model_version=model_version,
            source_recording_id=source_recording_id,
            source_start_ms=source_start_ms,
            source_end_ms=source_end_ms,
            created_at=datetime.now(timezone.utc),
            is_novel=is_novel,
        )

        if is_novel:
            self._reservoir_add(
                entry, self._novel_buffer, self._novel_capacity, self._novel_seen
            )
            self._novel_seen += 1
        else:
            self._reservoir_add(
                entry, self._common_buffer, self._common_capacity, self._common_seen
            )
            self._common_seen += 1

    def sample(self, batch_size: int) -> List[ReplayEntry]:
        """
        Sample a batch from the buffer.

        The batch is drawn proportionally from both the common and novel
        pools to ensure novel experiences are always represented.

        Args:
            batch_size: Number of entries to sample.

        Returns:
            List of ReplayEntry sampled from the buffer.
        """
        total = len(self._common_buffer) + len(self._novel_buffer)
        if total == 0:
            return []

        batch_size = min(batch_size, total)

        # Allocate proportionally, ensuring at least 1 from novel if available
        if self._novel_buffer:
            n_novel = max(1, int(batch_size * self.novelty_reserve_ratio))
            n_novel = min(n_novel, len(self._novel_buffer))
            n_common = batch_size - n_novel
            n_common = min(n_common, len(self._common_buffer))
        else:
            n_common = min(batch_size, len(self._common_buffer))
            n_novel = 0

        samples: List[ReplayEntry] = []
        if n_common > 0:
            samples.extend(random.sample(self._common_buffer, n_common))
        if n_novel > 0:
            samples.extend(random.sample(self._novel_buffer, n_novel))

        random.shuffle(samples)
        return samples

    def get_all_vectors(self) -> np.ndarray:
        """Return all buffered vectors as (N, D) matrix."""
        all_entries = self._common_buffer + self._novel_buffer
        if not all_entries:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([e.vector for e in all_entries], axis=0)

    @property
    def size(self) -> int:
        """Current total number of entries in the buffer."""
        return len(self._common_buffer) + len(self._novel_buffer)

    @property
    def total_seen(self) -> int:
        """Total number of experiences ever offered to the buffer."""
        return self._common_seen + self._novel_seen

    @staticmethod
    def _reservoir_add(
        entry: ReplayEntry,
        buffer: List[ReplayEntry],
        capacity: int,
        seen: int,
    ) -> None:
        """Standard reservoir sampling insertion."""
        if len(buffer) < capacity:
            buffer.append(entry)
        else:
            # Replace element at random index with probability capacity / (seen + 1)
            idx = random.randint(0, seen)
            if idx < capacity:
                buffer[idx] = entry
