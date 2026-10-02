"""
Relational Memory — Temporal Transition Graph.

Stores the directed, weighted graph of transitions between prototypes.
When speech is processed as a sequence of prototype matches:

    [Proto_A, Proto_A, Proto_B, Proto_C, Proto_A, ...]

this module records:
    - A → B (count, mean temporal delay)
    - B → C (count, mean temporal delay)
    - C → A (count, mean temporal delay)

This is not a clustering result — it is a learned temporal structure
that captures how speech patterns flow into each other.

The graph enables queries like:
    "What usually follows Pattern 17?"
    "How quickly does the system transition from Pattern 3 to Pattern 9?"
    "What is the most common 3-step sequence?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Edge:
    """A directed edge between two prototypes."""

    from_id: str
    to_id: str
    transition_count: int = 0
    total_delay_ms: float = 0.0

    @property
    def mean_delay_ms(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.total_delay_ms / self.transition_count


class RelationalMemory:
    """
    Directed transition graph between speech prototypes.

    Records how prototype activations flow over time. Each time the system
    processes a speech segment and matches it to prototypes, the resulting
    sequence of prototype IDs is fed to this graph to update transition
    counts and temporal delays.

    Internal structure:
        _adjacency[from_id][to_id] = Edge(...)
    """

    def __init__(self) -> None:
        self._adjacency: Dict[str, Dict[str, Edge]] = {}
        self._total_transitions: int = 0

    def record_transition(
        self,
        from_id: str,
        to_id: str,
        delay_ms: float = 0.0,
    ) -> Edge:
        """
        Record a single transition from one prototype to another.

        Args:
            from_id: Source prototype ID.
            to_id: Target prototype ID.
            delay_ms: Time delay between the two activations in milliseconds.

        Returns:
            The updated Edge.
        """
        if from_id not in self._adjacency:
            self._adjacency[from_id] = {}

        if to_id not in self._adjacency[from_id]:
            self._adjacency[from_id][to_id] = Edge(
                from_id=from_id, to_id=to_id
            )

        edge = self._adjacency[from_id][to_id]
        edge.transition_count += 1
        edge.total_delay_ms += delay_ms
        self._total_transitions += 1

        return edge

    def record_sequence(
        self,
        prototype_ids: List[str],
        frame_times_ms: Optional[List[float]] = None,
    ) -> List[Edge]:
        """
        Record a full sequence of prototype activations.

        Consecutive identical IDs are collapsed (self-transitions are not
        recorded by default, as they represent sustained activation
        of the same prototype, not a transition).

        Args:
            prototype_ids: Ordered list of prototype IDs from a speech segment.
            frame_times_ms: Optional timestamps (ms) for each prototype activation.

        Returns:
            List of updated edges.
        """
        if len(prototype_ids) < 2:
            return []

        # Collapse consecutive duplicates
        collapsed_ids: List[str] = [prototype_ids[0]]
        collapsed_times: List[float] = [
            frame_times_ms[0] if frame_times_ms else 0.0
        ]

        for i in range(1, len(prototype_ids)):
            if prototype_ids[i] != prototype_ids[i - 1]:
                collapsed_ids.append(prototype_ids[i])
                collapsed_times.append(
                    frame_times_ms[i] if frame_times_ms else 0.0
                )

        # Record edges
        edges: List[Edge] = []
        for i in range(len(collapsed_ids) - 1):
            delay = collapsed_times[i + 1] - collapsed_times[i]
            edge = self.record_transition(
                from_id=collapsed_ids[i],
                to_id=collapsed_ids[i + 1],
                delay_ms=max(0.0, delay),
            )
            edges.append(edge)

        return edges

    def get_successors(self, prototype_id: str) -> List[Tuple[str, int, float]]:
        """
        Get all prototypes that follow the given prototype.

        Returns:
            List of (successor_id, transition_count, mean_delay_ms)
            sorted by transition count descending.
        """
        if prototype_id not in self._adjacency:
            return []

        result = [
            (edge.to_id, edge.transition_count, edge.mean_delay_ms)
            for edge in self._adjacency[prototype_id].values()
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_predecessors(self, prototype_id: str) -> List[Tuple[str, int, float]]:
        """
        Get all prototypes that precede the given prototype.

        Returns:
            List of (predecessor_id, transition_count, mean_delay_ms)
            sorted by transition count descending.
        """
        result = []
        for src_id, neighbors in self._adjacency.items():
            if prototype_id in neighbors:
                edge = neighbors[prototype_id]
                result.append((src_id, edge.transition_count, edge.mean_delay_ms))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_edge(self, from_id: str, to_id: str) -> Optional[Edge]:
        """Get a specific edge, or None if it doesn't exist."""
        return self._adjacency.get(from_id, {}).get(to_id)

    def get_all_edges(self) -> List[Edge]:
        """Return all edges in the graph."""
        edges = []
        for neighbors in self._adjacency.values():
            edges.extend(neighbors.values())
        return edges

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def num_edges(self) -> int:
        return sum(len(n) for n in self._adjacency.values())

    @property
    def num_nodes(self) -> int:
        """Number of unique prototype IDs that appear in the graph."""
        nodes = set()
        for src, neighbors in self._adjacency.items():
            nodes.add(src)
            for dst in neighbors:
                nodes.add(dst)
        return len(nodes)

    def clear(self) -> None:
        """Clear all recorded transitions."""
        self._adjacency.clear()
        self._total_transitions = 0
