"""
Stability Tester — EWC-Inspired Probe Evaluation.

Before promoting a candidate encoder (θ') to replace the current
encoder (θ), the system must verify that the new model hasn't
suffered from catastrophic forgetting or representation collapse.

This module implements two complementary validation mechanisms:

1. Fixed Probe Evaluation:
   Maintain a frozen set of reference audio segments ("probes").
   After training θ', encode the probes with both θ and θ'.
   Compare the embeddings. If the cosine similarity drops below
   a threshold, the candidate has drifted too far.

2. Fisher Information Penalty (EWC-inspired):
   Track which weights are most important to the current model's
   performance using the diagonal of the Fisher Information Matrix.
   If θ' has changed those weights significantly, flag the candidate.

Algorithm Reference:
    Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting
    in neural networks." PNAS 114(13).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """
    Result of stability testing a candidate encoder.

    Attributes:
        is_stable: Whether the candidate passed all stability checks.
        mean_probe_similarity: Average cosine similarity between θ and θ'
                                on the fixed probes.
        min_probe_similarity: Worst-case probe similarity.
        weight_drift: Mean L2 distance of parameter changes weighted
                       by Fisher importance.
        rejection_reasons: List of reasons if the candidate was rejected.
        timestamp: When this test was performed.
    """

    is_stable: bool
    mean_probe_similarity: float
    min_probe_similarity: float
    weight_drift: float
    rejection_reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StabilityTester:
    """
    Validates candidate encoders before promotion.

    Usage:
        1. After the first encoder is loaded, call register_baseline()
           to snapshot the weights and compute Fisher importance.
        2. Periodically add probe audio with add_probe().
        3. When a candidate θ' is trained, call evaluate() to check stability.

    Parameters:
        similarity_threshold: Minimum acceptable mean cosine similarity
                               between current and candidate probe embeddings.
        min_similarity_threshold: Minimum acceptable WORST-CASE probe similarity.
        max_weight_drift: Maximum acceptable Fisher-weighted parameter drift.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_similarity_threshold: float = 0.70,
        max_weight_drift: float = 5.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.max_weight_drift = max_weight_drift

        # Baseline state
        self._baseline_params: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_diag: Optional[Dict[str, torch.Tensor]] = None

        # Probe embeddings from the current encoder
        self._probe_embeddings: List[np.ndarray] = []

    def register_baseline(self, model: nn.Module) -> None:
        """
        Snapshot the current encoder's parameters as the baseline.

        This should be called after the initial model is loaded
        and whenever a candidate is successfully promoted.

        Args:
            model: The current encoder model.
        """
        self._baseline_params = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
        }
        logger.info(
            "Baseline registered with %d parameter tensors.",
            len(self._baseline_params),
        )

    def compute_fisher(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
    ) -> None:
        """
        Estimate the diagonal of the Fisher Information Matrix.

        This tells us which weights are "important" to the current model.
        Computed by averaging the squared gradients over a set of inputs.

        Args:
            model: The current encoder model.
            sample_inputs: List of input waveform tensors for Fisher estimation.
        """
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        model.eval()

        for inp in sample_inputs:
            model.zero_grad()
            try:
                features, _ = model.extract_features(inp)
                # Use the last layer's mean as a pseudo-loss
                output = features[-1].mean()
                output.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher:
                        fisher[name] += param.grad.data ** 2
            except Exception as e:
                logger.warning("Fisher computation failed for a sample: %s", e)
                continue

        # Average over samples
        n = max(1, len(sample_inputs))
        for name in fisher:
            fisher[name] /= n

        self._fisher_diag = fisher
        logger.info("Fisher information computed over %d samples.", n)

    def add_probe_embedding(self, embedding: np.ndarray) -> None:
        """
        Add a probe embedding from the current encoder.

        These are the reference embeddings that will be compared
        against the candidate encoder's output.

        Args:
            embedding: A (D,) pooled embedding from a fixed probe audio.
        """
        self._probe_embeddings.append(embedding.copy())

    def evaluate(
        self,
        candidate_model: nn.Module,
        probe_inputs: List[torch.Tensor],
        extraction_layers: tuple = (4, 9),
    ) -> StabilityReport:
        """
        Evaluate a candidate encoder against the baseline.

        Args:
            candidate_model: The candidate encoder θ'.
            probe_inputs: The same probe audio tensors used to generate
                          the reference embeddings.
            extraction_layers: Layer range for embedding extraction.

        Returns:
            StabilityReport with pass/fail verdict and metrics.
        """
        rejection_reasons: List[str] = []

        # ── Probe Similarity Check ──────────────────────────────────
        if not self._probe_embeddings or not probe_inputs:
            logger.warning("No probes available for stability testing.")
            return StabilityReport(
                is_stable=True,
                mean_probe_similarity=1.0,
                min_probe_similarity=1.0,
                weight_drift=0.0,
                rejection_reasons=["No probes available — auto-pass."],
            )

        candidate_model.eval()
        similarities: List[float] = []

        with torch.no_grad():
            for i, inp in enumerate(probe_inputs):
                if i >= len(self._probe_embeddings):
                    break

                try:
                    features, _ = candidate_model.extract_features(inp)
                    layer_start, layer_end = extraction_layers
                    selected = torch.stack(features[layer_start:layer_end], dim=0)
                    candidate_emb = (
                        selected.mean(dim=0).squeeze(0).mean(dim=0).cpu().numpy()
                    )

                    reference_emb = self._probe_embeddings[i]
                    sim = self._cosine_similarity(reference_emb, candidate_emb)
                    similarities.append(sim)
                except Exception as e:
                    logger.warning("Probe %d evaluation failed: %s", i, e)
                    similarities.append(0.0)

        mean_sim = float(np.mean(similarities)) if similarities else 0.0
        min_sim = float(np.min(similarities)) if similarities else 0.0

        if mean_sim < self.similarity_threshold:
            rejection_reasons.append(
                f"Mean probe similarity {mean_sim:.4f} < "
                f"threshold {self.similarity_threshold}"
            )

        if min_sim < self.min_similarity_threshold:
            rejection_reasons.append(
                f"Min probe similarity {min_sim:.4f} < "
                f"threshold {self.min_similarity_threshold}"
            )

        # ── Fisher-Weighted Drift Check ─────────────────────────────
        weight_drift = 0.0
        if self._baseline_params is not None:
            candidate_params = {
                name: param.data.cpu()
                for name, param in candidate_model.named_parameters()
            }

            total_penalty = 0.0
            n_params = 0

            for name, baseline_val in self._baseline_params.items():
                if name not in candidate_params:
                    continue

                delta = candidate_params[name] - baseline_val

                if self._fisher_diag is not None and name in self._fisher_diag:
                    fisher_weight = self._fisher_diag[name].cpu()
                    penalty = (fisher_weight * delta ** 2).sum().item()
                else:
                    penalty = (delta ** 2).sum().item()

                total_penalty += penalty
                n_params += 1

            weight_drift = total_penalty / max(1, n_params)

            if weight_drift > self.max_weight_drift:
                rejection_reasons.append(
                    f"Fisher-weighted drift {weight_drift:.4f} > "
                    f"threshold {self.max_weight_drift}"
                )

        is_stable = len(rejection_reasons) == 0

        return StabilityReport(
            is_stable=is_stable,
            mean_probe_similarity=mean_sim,
            min_probe_similarity=min_sim,
            weight_drift=weight_drift,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        return float(dot / norm)
