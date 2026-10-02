"""
Episodic Memory.

An append-only, immutable log of every speech experience the system
has ever processed. Each episode records:
    - The pooled embedding vector (for retrieval)
    - The full temporal sequence shape metadata
    - A reference to the source audio segment
    - The encoder version that produced it
    - The timestamp of when it was experienced

Episodic memory is the raw "autobiography" of the system.
It is never modified or deleted — only appended to.

Storage:
    In-memory list for now. Designed to be backed by a database
    (PostgreSQL + pgvector) by implementing the EpisodicStore protocol.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Episode:
    """A single, immutable speech experience."""

    id: str
    pooled_vector: np.ndarray       # (D,) — for retrieval and prototype matching
    sequence_length: int            # T — number of temporal frames
    embedding_dim: int              # D — vector dimension
    model_version: str              # Which encoder produced this
    source_recording_id: Optional[str]
    source_start_ms: int
    source_end_ms: int
    created_at: datetime

    @property
    def duration_ms(self) -> int:
        return self.source_end_ms - self.source_start_ms


class EpisodicStore(Protocol):
    """
    Protocol for episodic memory backends.

    Implement this to swap the in-memory store for PostgreSQL/pgvector.
    """

    def append(self, episode: Episode) -> None: ...
    def get_recent(self, n: int) -> List[Episode]: ...
    def get_all_vectors(self) -> np.ndarray: ...
    def count(self) -> int: ...


class EpisodicMemory:
    """
    In-memory episodic memory store.

    Append-only log of all speech experiences. Supports:
        - Appending new episodes
        - Retrieving recent episodes
        - Bulk vector export for structure discovery

    Thread-safety: Not thread-safe. Use external synchronization
    if accessed from multiple threads.
    """

    def __init__(self) -> None:
        self._episodes: List[Episode] = []
        self._vectors: List[np.ndarray] = []

    def record(
        self,
        pooled_vector: np.ndarray,
        sequence_length: int,
        embedding_dim: int,
        model_version: str,
        source_recording_id: Optional[str] = None,
        source_start_ms: int = 0,
        source_end_ms: int = 0,
    ) -> Episode:
        """
        Record a new speech experience.

        Args:
            pooled_vector: The (D,) pooled embedding from the encoder.
            sequence_length: Number of temporal frames T.
            embedding_dim: Embedding dimension D.
            model_version: Encoder version string.
            source_recording_id: Optional link to the source recording.
            source_start_ms: Start offset in the source recording.
            source_end_ms: End offset in the source recording.

        Returns:
            The created Episode.
        """
        episode = Episode(
            id=str(uuid.uuid4()),
            pooled_vector=pooled_vector.copy(),
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            model_version=model_version,
            source_recording_id=source_recording_id,
            source_start_ms=source_start_ms,
            source_end_ms=source_end_ms,
            created_at=datetime.now(timezone.utc),
        )
        self._episodes.append(episode)
        self._vectors.append(pooled_vector.copy())
        return episode

    def get_recent(self, n: int) -> List[Episode]:
        """Return the N most recent episodes."""
        return self._episodes[-n:]

    def get_all_vectors(self) -> np.ndarray:
        """
        Return all pooled vectors as a (N, D) matrix.
        Used by the global structure discovery engine.
        """
        if not self._vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self._vectors, axis=0)

    def get_all_episodes(self) -> List[Episode]:
        """Return all episodes in chronological order."""
        return list(self._episodes)

    def count(self) -> int:
        """Total number of recorded episodes."""
        return len(self._episodes)

    def get_episodes_since(self, since: datetime) -> List[Episode]:
        """Return episodes recorded after the given timestamp."""
        return [e for e in self._episodes if e.created_at > since]
