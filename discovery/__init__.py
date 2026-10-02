"""
Structure Discovery Engine — Subsystem C.

Responsible for finding order in the embedding space:
    - NoveltyDetector:          Online, per-input novelty scoring
    - GlobalStructureDiscovery: Periodic batch UMAP + HDBSCAN clustering
"""

from .novelty import NoveltyDetector, NoveltyEvent
from .global_structure import GlobalStructureDiscovery, DiscoveryResult

__all__ = [
    "NoveltyDetector",
    "NoveltyEvent",
    "GlobalStructureDiscovery",
    "DiscoveryResult",
]
