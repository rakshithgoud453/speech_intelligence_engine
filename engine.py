"""
Speech Learning Engine — The Orchestrator.

This is the central coordinator that ties all four subsystems together
and executes the Continual Learning Loop described in CONVERSATION_CONTEXT_2.md.

The loop, per speech input:
    1. Segment audio into voiced regions.
    2. Encode each segment with the current encoder θ → temporal sequence.
    3. Pool the sequence into a single vector for memory operations.
    4. Store the experience in Episodic Memory (immutable).
    5. Update Prototype Memory (GNG) — nearest match or novelty.
    6. Record temporal transitions in Relational Memory.
    7. Add to Replay Buffer (reservoir sampling).
    8. Track novelty statistics.

Periodically (triggered manually or by threshold):
    9.  Run Global Structure Discovery (UMAP + HDBSCAN).
    10. Reconcile discovered structures with prototype memory.
    11. Train a candidate encoder θ' using replay buffer.
    12. Test θ' for stability (EWC probes).
    13. Promote θ' → θ if stable, reject otherwise.

This module is the ONLY entry point for external code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa

from .representation import SpeechEncoder, VoiceActivitySegmenter, Segment
from .representation.encoder import TemporalEmbedding
from .memory import (
    EpisodicMemory,
    Episode,
    PrototypeMemory,
    Prototype,
    RelationalMemory,
    ReplayBuffer,
)
from .discovery import NoveltyDetector, GlobalStructureDiscovery, DiscoveryResult
from .learning import StabilityTester, StabilityReport

logger = logging.getLogger(__name__)


@dataclass
class AssimilationResult:
    """Result of assimilating a single audio file or array."""

    recording_id: Optional[str]
    num_segments: int
    num_episodes: int
    num_novel: int
    prototype_matches: List[str]
    novelty_rate: float
    duration_sec: float


@dataclass
class EngineState:
    """Snapshot of the engine's current internal state."""

    total_episodes: int
    total_prototypes: int
    total_transitions: int
    total_graph_edges: int
    replay_buffer_size: int
    replay_buffer_total_seen: int
    overall_novelty_rate: float
    recent_novelty_rate: float
    is_novelty_burst: bool
    encoder_version: str


class SpeechLearningEngine:
    """
    The central orchestrator for the Speech Intelligence Learning System.

    This is designed to be used as a long-running stateful object.
    Feed it audio, and it learns. Query it, and it reports what it knows.

    Parameters:
        model_type: Foundation model to use ('hubert', 'wavlm', 'wav2vec2').
        device: PyTorch device. Auto-detected if None.
        replay_capacity: Size of the replay buffer.
        max_prototypes: Maximum number of GNG prototype nodes.
        discovery_min_episodes: Minimum episodes before global discovery runs.
    """

    def __init__(
        self,
        model_type: str = "hubert",
        device: Optional[str] = None,
        replay_capacity: int = 5000,
        max_prototypes: int = 500,
        discovery_min_episodes: int = 100,
    ):
        # ── Subsystem A: Representation ─────────────────────────────
        self.encoder = SpeechEncoder(
            model_type=model_type, device=device
        )
        self.segmenter = VoiceActivitySegmenter(sample_rate=16000)

        # ── Subsystem B: Memory ─────────────────────────────────────
        self.episodic_memory = EpisodicMemory()
        self.prototype_memory = PrototypeMemory(
            embedding_dim=768,
            max_nodes=max_prototypes,
        )
        self.relational_memory = RelationalMemory()
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # ── Subsystem C: Discovery ──────────────────────────────────
        self.novelty_detector = NoveltyDetector()
        self.global_discovery = GlobalStructureDiscovery(
            min_vectors_for_discovery=discovery_min_episodes,
        )

        # ── Subsystem D: Learning ───────────────────────────────────
        self.stability_tester = StabilityTester()

        # Register baseline weights for stability testing
        self.stability_tester.register_baseline(self.encoder.get_model())

        logger.info(
            "SpeechLearningEngine initialized with encoder=%s, device=%s",
            model_type,
            self.encoder.device,
        )

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API: Assimilate (the main learning loop)
    # ═══════════════════════════════════════════════════════════════

    def assimilate(
        self,
        audio_input: str | np.ndarray,
        sr: Optional[int] = None,
        recording_id: Optional[str] = None,
    ) -> AssimilationResult:
        """
        Assimilate a speech recording into the system's knowledge.

        This is the primary entry point. The system:
            1. Segments the audio.
            2. Encodes each segment.
            3. Updates all memory structures.
            4. Tracks novelty.

        The system does NOT return predictions or classifications.
        It learns.

        Args:
            audio_input: Path to an audio file, or a 1-D float32 numpy array.
            sr: Sample rate (required if audio_input is an array).
            recording_id: Optional external recording identifier.

        Returns:
            AssimilationResult with statistics about what happened.
        """
        # Load audio
        waveform, sample_rate = self._load_audio(audio_input, sr)
        duration_sec = len(waveform) / sample_rate

        # Step 1: Segment into voiced regions
        segments = self.segmenter.segment(waveform, sr=sample_rate)

        if not segments:
            logger.info("No voiced segments found in audio.")
            return AssimilationResult(
                recording_id=recording_id,
                num_segments=0,
                num_episodes=0,
                num_novel=0,
                prototype_matches=[],
                novelty_rate=0.0,
                duration_sec=duration_sec,
            )

        logger.info(
            "Segmented audio into %d voiced regions (%.1fs total).",
            len(segments),
            duration_sec,
        )

        # Process each segment through the learning loop
        num_novel = 0
        all_proto_ids: List[str] = []
        all_frame_times_ms: List[float] = []

        for segment in segments:
            # Step 2: Encode
            embedding = self.encoder.encode(
                segment.waveform, sr=segment.sample_rate
            )

            # Step 3: Pool for memory operations
            pooled = embedding.pooled()

            # Step 4: Episodic memory (immutable log)
            episode = self.episodic_memory.record(
                pooled_vector=pooled,
                sequence_length=embedding.num_frames,
                embedding_dim=embedding.embedding_dim,
                model_version=embedding.model_version,
                source_recording_id=recording_id,
                source_start_ms=segment.start_ms,
                source_end_ms=segment.end_ms,
            )

            # Step 5: Prototype memory (GNG update)
            nearest_proto, is_novel = self.prototype_memory.update(pooled)
            distance = nearest_proto.distance_to(pooled)

            # Step 6: Novelty tracking
            self.novelty_detector.record(
                vector=pooled,
                is_novel=is_novel,
                distance=distance,
                nearest_prototype_id=nearest_proto.id,
            )
            if is_novel:
                num_novel += 1

            # Step 7: Replay buffer (reservoir sampling)
            self.replay_buffer.add(
                vector=pooled,
                model_version=embedding.model_version,
                is_novel=is_novel,
                source_recording_id=recording_id,
                source_start_ms=segment.start_ms,
                source_end_ms=segment.end_ms,
            )

            # Collect prototype sequence for relational memory
            all_proto_ids.append(nearest_proto.id)
            all_frame_times_ms.append(float(segment.start_ms))

        # Step 8: Relational memory — record the prototype transition sequence
        if len(all_proto_ids) > 1:
            self.relational_memory.record_sequence(
                prototype_ids=all_proto_ids,
                frame_times_ms=all_frame_times_ms,
            )

        result = AssimilationResult(
            recording_id=recording_id,
            num_segments=len(segments),
            num_episodes=len(segments),
            num_novel=num_novel,
            prototype_matches=[pid for pid in all_proto_ids],
            novelty_rate=num_novel / len(segments) if segments else 0.0,
            duration_sec=duration_sec,
        )

        logger.info(
            "Assimilation complete: %d segments, %d novel, "
            "%d prototypes active, novelty rate=%.2f",
            result.num_segments,
            result.num_novel,
            self.prototype_memory.num_prototypes,
            result.novelty_rate,
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API: Discovery (periodic batch operation)
    # ═══════════════════════════════════════════════════════════════

    def run_discovery(self) -> Optional[DiscoveryResult]:
        """
        Run global structure discovery (UMAP + HDBSCAN) on all episodes.

        This is a heavyweight operation. Call it periodically
        (e.g., after every N assimilations or on a schedule).

        Returns:
            DiscoveryResult with cluster assignments, or None if
            there aren't enough episodes yet.
        """
        vectors = self.episodic_memory.get_all_vectors()

        result = self.global_discovery.discover(vectors)

        if result is not None:
            # Reconcile: update prototype memory with discovered centroids
            new_prototypes = []
            for label, centroid in result.centroids.items():
                proto = Prototype(
                    id=f"discovered_{label}",
                    vector=centroid,
                    count=result.cluster_sizes.get(label, 0),
                )
                new_prototypes.append(proto)

            if new_prototypes:
                self.prototype_memory.replace_prototypes(new_prototypes)
                logger.info(
                    "Prototype memory updated with %d discovered structures.",
                    len(new_prototypes),
                )

            # Clear the novelty buffer since we've just re-analyzed everything
            self.novelty_detector.clear_buffer()

        return result

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API: State Inspection
    # ═══════════════════════════════════════════════════════════════

    def get_state(self) -> EngineState:
        """
        Get a snapshot of the engine's current internal state.

        This is the "what does the system know" query.
        """
        return EngineState(
            total_episodes=self.episodic_memory.count(),
            total_prototypes=self.prototype_memory.num_prototypes,
            total_transitions=self.relational_memory.total_transitions,
            total_graph_edges=self.relational_memory.num_edges,
            replay_buffer_size=self.replay_buffer.size,
            replay_buffer_total_seen=self.replay_buffer.total_seen,
            overall_novelty_rate=self.novelty_detector.novelty_rate,
            recent_novelty_rate=self.novelty_detector.recent_novelty_rate,
            is_novelty_burst=self.novelty_detector.is_burst,
            encoder_version=self.encoder.version,
        )

    def get_prototypes(self) -> List[Prototype]:
        """Return all current prototypes."""
        return self.prototype_memory.prototypes

    def get_prototype_graph(self) -> Dict[str, Any]:
        """
        Return the relational memory as a serializable graph structure.

        Returns:
            Dictionary with 'nodes' (prototype IDs and counts) and
            'edges' (transitions with counts and delays).
        """
        nodes = []
        for proto in self.prototype_memory.prototypes:
            successors = self.relational_memory.get_successors(proto.id)
            nodes.append({
                "id": proto.id,
                "count": proto.count,
                "first_seen": proto.first_seen.isoformat(),
                "last_seen": proto.last_seen.isoformat(),
                "successors": [
                    {"to": s[0], "count": s[1], "mean_delay_ms": s[2]}
                    for s in successors
                ],
            })

        edges = []
        for edge in self.relational_memory.get_all_edges():
            edges.append({
                "from": edge.from_id,
                "to": edge.to_id,
                "count": edge.transition_count,
                "mean_delay_ms": edge.mean_delay_ms,
            })

        return {"nodes": nodes, "edges": edges}

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: Audio Loading
    # ═══════════════════════════════════════════════════════════════

    def _load_audio(
        self,
        audio_input: str | np.ndarray,
        sr: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        """Load and normalize audio to 16kHz mono float32."""
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            waveform, file_sr = librosa.load(
                audio_input, sr=16000, mono=True
            )
            return waveform.astype(np.float32), 16000

        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)
            if sr is not None and sr != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sr, target_sr=16000
                )
            return waveform.astype(np.float32), 16000

        else:
            raise TypeError(
                "audio_input must be a file path or numpy array."
            )
