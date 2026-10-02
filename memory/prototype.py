"""
Prototype Memory — Growing Neural Gas (GNG).

Implements an online, incremental topology-learning algorithm that
discovers and maintains prototype nodes in the embedding space.

Each prototype represents a recurring region of the latent space.
Prototypes do NOT have human semantic labels — they are simply
"Pattern 0", "Pattern 1", etc. The system discovers meaning through
structure, not through labeling.

Algorithm Reference:
    Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies."
    Advances in Neural Information Processing Systems 7 (NIPS 1994).

The GNG algorithm:
    1. Start with two random nodes.
    2. For each input signal:
       a. Find the two nearest nodes (winner s1, runner-up s2).
       b. Increment the age of all edges emanating from s1.
       c. Add the squared distance to s1's cumulative error.
       d. Move s1 and its topological neighbors toward the input.
       e. Create an edge between s1 and s2 (or reset its age to 0).
       f. Remove edges older than max_age. Remove isolated nodes.
    3. Every λ steps, insert a new node between the node with
       the highest error and its neighbor with the highest error.
    4. Decay all errors by a factor d.

This is NOT a toy implementation — it is the standard GNG algorithm
as described in the original paper, adapted for high-dimensional
speech embedding vectors.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Prototype:
    """
    A single prototype node in the Growing Neural Gas network.

    Attributes:
        id: Unique identifier for this prototype.
        vector: The prototype's position in embedding space (D,).
        error: Accumulated quantization error.
        count: Number of times this prototype was the winner.
        first_seen: When this prototype was created.
        last_seen: When this prototype was last activated.
    """

    id: str
    vector: np.ndarray
    error: float = 0.0
    count: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def distance_to(self, x: np.ndarray) -> float:
        """Euclidean distance from this prototype to vector x."""
        return float(np.linalg.norm(self.vector - x))


class PrototypeMemory:
    """
    Growing Neural Gas network for online prototype discovery.

    This is the system's "concept formation" mechanism. As the system
    processes more speech, prototypes are born, move, and die based
    on the structure of the incoming embedding space.

    Parameters:
        embedding_dim: Dimensionality of the embedding vectors (e.g., 768).
        max_age: Maximum age of an edge before removal. Controls topology freshness.
        lambda_interval: Insert a new node every λ input signals.
        epsilon_winner: Learning rate for the winning node.
        epsilon_neighbor: Learning rate for the winner's neighbors.
        error_decay: Multiplicative decay factor for accumulated errors.
        max_nodes: Maximum number of prototype nodes allowed.
        novelty_threshold: Distance threshold above which an input is flagged as novel.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_age: int = 50,
        lambda_interval: int = 100,
        epsilon_winner: float = 0.05,
        epsilon_neighbor: float = 0.006,
        error_decay: float = 0.995,
        max_nodes: int = 500,
        novelty_threshold: float | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.max_age = max_age
        self.lambda_interval = lambda_interval
        self.epsilon_winner = epsilon_winner
        self.epsilon_neighbor = epsilon_neighbor
        self.error_decay = error_decay
        self.max_nodes = max_nodes
        self.novelty_threshold = novelty_threshold

        # Internal state
        self._nodes: Dict[str, Prototype] = {}
        self._edges: Dict[str, Dict[str, int]] = {}  # node_id -> {neighbor_id: age}
        self._step_count: int = 0
        self._initialized: bool = False

        # Adaptive novelty threshold tracking
        self._distance_history: List[float] = []
        self._distance_window: int = 500

    @property
    def prototypes(self) -> List[Prototype]:
        """All current prototype nodes."""
        return list(self._nodes.values())

    @property
    def num_prototypes(self) -> int:
        return len(self._nodes)

    def update(self, x: np.ndarray) -> Tuple[Prototype, bool]:
        """
        Process a single input vector through the GNG algorithm.

        This is the core online learning step. For every speech embedding
        the system encounters, this method:
            1. Finds the nearest prototype
            2. Updates the topology
            3. Periodically inserts new prototypes
            4. Detects novelty

        Args:
            x: Input embedding vector of shape (D,).

        Returns:
            Tuple of (nearest_prototype, is_novel).
            is_novel is True if the input is far from all known prototypes.
        """
        x = x.astype(np.float32)

        # Bootstrap: need at least 2 nodes to start GNG
        if not self._initialized or len(self._nodes) < 2:
            return self._bootstrap(x)

        self._step_count += 1

        # Step 1: Find the two nearest nodes
        s1, dist1, s2, dist2 = self._find_two_nearest(x)

        # Step 2: Track distance for adaptive novelty threshold
        self._distance_history.append(dist1)
        if len(self._distance_history) > self._distance_window:
            self._distance_history.pop(0)

        # Step 3: Determine novelty
        is_novel = self._is_novel(dist1)

        # Step 4: Increment age of all edges from the winner
        if s1.id in self._edges:
            for neighbor_id in self._edges[s1.id]:
                self._edges[s1.id][neighbor_id] += 1

        # Step 5: Accumulate error on the winner
        s1.error += dist1 ** 2
        s1.count += 1
        s1.last_seen = datetime.now(timezone.utc)

        # Step 6: Move winner toward input
        s1.vector += self.epsilon_winner * (x - s1.vector)

        # Step 7: Move winner's neighbors toward input
        if s1.id in self._edges:
            for neighbor_id in list(self._edges[s1.id].keys()):
                if neighbor_id in self._nodes:
                    neighbor = self._nodes[neighbor_id]
                    neighbor.vector += self.epsilon_neighbor * (x - neighbor.vector)

        # Step 8: Create or refresh edge between s1 and s2
        self._set_edge(s1.id, s2.id, age=0)

        # Step 9: Remove old edges and isolated nodes
        self._prune_edges()
        self._remove_isolated_nodes()

        # Step 10: Every λ steps, insert a new node
        if self._step_count % self.lambda_interval == 0:
            self._insert_node()

        # Step 11: Decay all errors
        for node in self._nodes.values():
            node.error *= self.error_decay

        return s1, is_novel

    def nearest(self, x: np.ndarray, k: int = 1) -> List[Tuple[Prototype, float]]:
        """
        Find the k nearest prototypes to vector x.

        Returns:
            List of (prototype, distance) pairs sorted by distance.
        """
        if not self._nodes:
            return []

        distances = [
            (node, node.distance_to(x))
            for node in self._nodes.values()
        ]
        distances.sort(key=lambda pair: pair[1])
        return distances[:k]

    def get_topology(self) -> List[Tuple[str, str]]:
        """Return all edges as (node_id_a, node_id_b) pairs."""
        edges = set()
        for src, neighbors in self._edges.items():
            for dst in neighbors:
                edge = tuple(sorted([src, dst]))
                edges.add(edge)
        return list(edges)

    def replace_prototypes(self, new_prototypes: List[Prototype]) -> None:
        """
        Replace all prototypes with a new set.
        Used by the Global Structure Discovery engine after HDBSCAN.
        """
        self._nodes.clear()
        self._edges.clear()
        for proto in new_prototypes:
            self._nodes[proto.id] = proto
            self._edges[proto.id] = {}
        self._initialized = len(self._nodes) >= 2

    # ─── Private GNG Methods ──────────────────────────────────────────

    def _bootstrap(self, x: np.ndarray) -> Tuple[Prototype, bool]:
        """Add initial nodes to bootstrap the GNG."""
        node = Prototype(
            id=str(uuid.uuid4()),
            vector=x.copy(),
            count=1,
        )
        self._nodes[node.id] = node
        self._edges[node.id] = {}

        if len(self._nodes) >= 2:
            self._initialized = True

        return node, True  # First experiences are always novel

    def _find_two_nearest(
        self, x: np.ndarray
    ) -> Tuple[Prototype, float, Prototype, float]:
        """Find the two nearest prototype nodes to input x."""
        distances = [
            (node, node.distance_to(x))
            for node in self._nodes.values()
        ]
        distances.sort(key=lambda pair: pair[1])
        s1, d1 = distances[0]
        s2, d2 = distances[1]
        return s1, d1, s2, d2

    def _is_novel(self, distance: float) -> bool:
        """Determine if the distance indicates a novel input."""
        if self.novelty_threshold is not None:
            return distance > self.novelty_threshold

        # Adaptive threshold: 2 standard deviations above the mean
        if len(self._distance_history) < 20:
            return True  # Not enough data yet, everything is novel

        arr = np.array(self._distance_history)
        threshold = float(np.mean(arr) + 2.0 * np.std(arr))
        return distance > threshold

    def _set_edge(self, a: str, b: str, age: int = 0) -> None:
        """Create or update an edge between nodes a and b."""
        if a not in self._edges:
            self._edges[a] = {}
        if b not in self._edges:
            self._edges[b] = {}
        self._edges[a][b] = age
        self._edges[b][a] = age

    def _prune_edges(self) -> None:
        """Remove edges older than max_age."""
        for src in list(self._edges.keys()):
            if src not in self._edges:
                continue
            expired = [
                dst
                for dst, age in self._edges[src].items()
                if age > self.max_age
            ]
            for dst in expired:
                del self._edges[src][dst]
                if dst in self._edges and src in self._edges[dst]:
                    del self._edges[dst][src]

    def _remove_isolated_nodes(self) -> None:
        """Remove nodes with no edges (disconnected from the topology)."""
        # Don't remove if we're still bootstrapping
        if len(self._nodes) <= 2:
            return

        isolated = [
            nid
            for nid, neighbors in self._edges.items()
            if len(neighbors) == 0
        ]
        for nid in isolated:
            if nid in self._nodes:
                del self._nodes[nid]
            if nid in self._edges:
                del self._edges[nid]

    def _insert_node(self) -> None:
        """
        Insert a new node between the node with the highest error
        and its neighbor with the highest error (GNG node insertion).
        """
        if len(self._nodes) >= self.max_nodes:
            return

        if not self._nodes:
            return

        # Find the node with the highest accumulated error
        q = max(self._nodes.values(), key=lambda n: n.error)

        # Find q's neighbor with the highest error
        if q.id not in self._edges or not self._edges[q.id]:
            return

        neighbor_ids = list(self._edges[q.id].keys())
        neighbors = [
            self._nodes[nid]
            for nid in neighbor_ids
            if nid in self._nodes
        ]
        if not neighbors:
            return

        f = max(neighbors, key=lambda n: n.error)

        # Create new node at the midpoint
        new_vector = 0.5 * (q.vector + f.vector)
        new_node = Prototype(
            id=str(uuid.uuid4()),
            vector=new_vector,
        )
        self._nodes[new_node.id] = new_node
        self._edges[new_node.id] = {}

        # Connect the new node to q and f
        self._set_edge(new_node.id, q.id, age=0)
        self._set_edge(new_node.id, f.id, age=0)

        # Remove the direct edge between q and f
        if q.id in self._edges and f.id in self._edges[q.id]:
            del self._edges[q.id][f.id]
        if f.id in self._edges and q.id in self._edges[f.id]:
            del self._edges[f.id][q.id]

        # Reduce error of q and f
        q.error *= 0.5
        f.error *= 0.5
        new_node.error = q.error
