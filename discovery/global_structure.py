"""
Global Structure Discovery — UMAP + HDBSCAN Pipeline.

A periodic, batch process that analyzes the full episodic memory
to discover macro-level structure that the online prototype learning
(GNG) may have missed.

Pipeline:
    1. Collect all pooled embedding vectors from episodic memory.
    2. Reduce dimensionality with UMAP (768-d → ~10-d).
       UMAP preserves manifold structure better than PCA for
       high-dimensional embedding spaces.
    3. Run HDBSCAN on the reduced space to find density-based clusters.
       HDBSCAN does not require specifying k (number of clusters).
       It discovers them from the data's density structure.
    4. Produce new prototype centroids from the discovered clusters.
    5. Report: new prototypes born, merged, or retired.

Algorithm References:
    - McInnes, L., Healy, J. (2017). "Accelerated Hierarchical Density
      Based Clustering." IEEE ICDMW.
    - McInnes, L., Healy, J., Melville, J. (2018). "UMAP: Uniform
      Manifold Approximation and Projection for Dimension Reduction."

Dependencies:
    pip install umap-learn hdbscan
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """
    Output of a global structure discovery run.

    Attributes:
        num_clusters: Number of clusters discovered.
        noise_count: Number of points classified as noise by HDBSCAN.
        cluster_sizes: Number of members in each cluster.
        centroids: Dictionary mapping cluster label → centroid vector.
        labels: Cluster label for each input vector (-1 = noise).
        timestamp: When this discovery was run.
    """

    num_clusters: int
    noise_count: int
    cluster_sizes: Dict[int, int]
    centroids: Dict[int, np.ndarray]
    labels: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GlobalStructureDiscovery:
    """
    Periodic batch structure discovery using UMAP → HDBSCAN.

    This is a heavyweight operation that should be run periodically
    (e.g., after every 1000 new episodes, or nightly), not on every input.

    Parameters:
        umap_n_components: Target dimensionality for UMAP reduction.
        umap_n_neighbors: UMAP locality parameter. Higher values capture
                          more global structure.
        umap_metric: Distance metric for UMAP.
        hdbscan_min_cluster_size: Minimum number of points to form a cluster.
                                   Higher = fewer, more stable clusters.
        hdbscan_min_samples: HDBSCAN conservatism. Higher = more noise points.
        min_vectors_for_discovery: Don't run discovery if fewer than this
                                    many vectors are available.
    """

    def __init__(
        self,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 15,
        umap_metric: str = "cosine",
        hdbscan_min_cluster_size: int = 20,
        hdbscan_min_samples: int = 5,
        min_vectors_for_discovery: int = 100,
    ):
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.min_vectors_for_discovery = min_vectors_for_discovery

        self._last_result: Optional[DiscoveryResult] = None

    def discover(self, vectors: np.ndarray) -> Optional[DiscoveryResult]:
        """
        Run the full UMAP → HDBSCAN discovery pipeline.

        Args:
            vectors: (N, D) matrix of pooled embedding vectors from episodic memory.

        Returns:
            DiscoveryResult with cluster assignments and centroids,
            or None if there aren't enough vectors.
        """
        if vectors.shape[0] < self.min_vectors_for_discovery:
            logger.info(
                "Skipping discovery: only %d vectors (need %d).",
                vectors.shape[0],
                self.min_vectors_for_discovery,
            )
            return None

        try:
            import umap
            import hdbscan
        except ImportError as e:
            logger.error(
                "Global discovery requires 'umap-learn' and 'hdbscan'. "
                "Install with: pip install umap-learn hdbscan. Error: %s",
                e,
            )
            return None

        logger.info(
            "Running global structure discovery on %d vectors (D=%d)...",
            vectors.shape[0],
            vectors.shape[1],
        )

        # Step 1: UMAP dimensionality reduction
        n_components = min(self.umap_n_components, vectors.shape[1])
        n_neighbors = min(self.umap_n_neighbors, vectors.shape[0] - 1)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=self.umap_metric,
            random_state=42,
        )
        reduced = reducer.fit_transform(vectors)

        logger.info("UMAP reduction complete: (%d, %d) → (%d, %d)",
                     vectors.shape[0], vectors.shape[1],
                     reduced.shape[0], reduced.shape[1])

        # Step 2: HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)

        # Step 3: Compute centroids in the ORIGINAL high-dimensional space
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        centroids: Dict[int, np.ndarray] = {}
        cluster_sizes: Dict[int, int] = {}

        for label in unique_labels:
            mask = labels == label
            cluster_vectors = vectors[mask]
            centroids[label] = cluster_vectors.mean(axis=0).astype(np.float32)
            cluster_sizes[label] = int(mask.sum())

        noise_count = int((labels == -1).sum())

        result = DiscoveryResult(
            num_clusters=len(unique_labels),
            noise_count=noise_count,
            cluster_sizes=cluster_sizes,
            centroids=centroids,
            labels=labels,
        )

        self._last_result = result

        logger.info(
            "Discovery complete: %d clusters found, %d noise points.",
            result.num_clusters,
            result.noise_count,
        )

        return result

    @property
    def last_result(self) -> Optional[DiscoveryResult]:
        """The result of the most recent discovery run."""
        return self._last_result
