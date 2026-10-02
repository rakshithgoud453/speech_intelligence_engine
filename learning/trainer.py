"""
Candidate Model Trainer.

Trains a new version of the encoder (θ') using the replay buffer
and the self-supervised masked prediction objective.

The training loop:
    1. Clone the current encoder θ → candidate θ'.
    2. For each batch from the replay buffer:
       a. Load the source audio from the replay entry references.
       b. Encode with the frozen teacher (θ) to get target representations.
       c. Encode with the student (θ') with masked frames.
       d. Compute masked prediction loss + EWC regularization.
       e. Update θ' via gradient descent.
    3. After training, pass θ' to the StabilityTester for validation.

This module does NOT handle the audio loading — it expects
pre-loaded waveform tensors. The orchestrator (engine.py) is
responsible for resolving replay buffer references to actual audio.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .objectives import SelfSupervisedObjective

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Output of a candidate training run.

    Attributes:
        candidate_state_dict: The trained candidate's weights.
        final_loss: Final training loss value.
        loss_history: Loss at each training step.
        num_steps: Total training steps completed.
        version: Version string for the candidate.
        timestamp: When training completed.
    """

    candidate_state_dict: dict
    final_loss: float
    loss_history: List[float]
    num_steps: int
    version: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CandidateTrainer:
    """
    Trains a candidate encoder using self-supervised replay.

    Parameters:
        learning_rate: Learning rate for the candidate optimizer.
        num_steps: Number of gradient update steps per training run.
        ewc_lambda: Strength of the EWC regularization penalty.
                    Higher = more conservative (less forgetting, less plasticity).
        extraction_layers: Which transformer layers to use for the objective.
        device: PyTorch device.
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        num_steps: int = 200,
        ewc_lambda: float = 1000.0,
        extraction_layers: Tuple[int, int] = (4, 9),
        device: Optional[str] = None,
    ):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.ewc_lambda = ewc_lambda
        self.extraction_layers = extraction_layers

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def train(
        self,
        teacher_model: nn.Module,
        waveforms: List[torch.Tensor],
        baseline_params: Optional[Dict[str, torch.Tensor]] = None,
        fisher_diag: Optional[Dict[str, torch.Tensor]] = None,
        current_version: str = "v0",
    ) -> TrainingResult:
        """
        Train a candidate encoder θ'.

        Args:
            teacher_model: The current encoder θ (used as frozen teacher).
            waveforms: List of waveform tensors from the replay buffer.
                       Each tensor should be shape (1, num_samples).
            baseline_params: Baseline parameter snapshot for EWC penalty.
            fisher_diag: Fisher information diagonal for EWC penalty.
            current_version: Version string of the current encoder.

        Returns:
            TrainingResult containing the candidate's state dict and metrics.
        """
        if not waveforms:
            raise ValueError("No waveforms provided for training.")

        # Clone the teacher to create the student (candidate)
        student_model = copy.deepcopy(teacher_model).to(self.device)
        student_model.train()

        teacher_model = teacher_model.to(self.device)
        teacher_model.eval()

        # Determine embedding dimension from teacher
        with torch.no_grad():
            test_features, _ = teacher_model.extract_features(
                waveforms[0].to(self.device)
            )
            embedding_dim = test_features[0].shape[-1]

        # Create the self-supervised objective
        objective = SelfSupervisedObjective(
            embedding_dim=embedding_dim
        ).to(self.device)

        # Optimizer for student + projection head
        optimizer = optim.AdamW(
            list(student_model.parameters()) + list(objective.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        loss_history: List[float] = []
        layer_start, layer_end = self.extraction_layers

        for step in range(self.num_steps):
            # Cycle through waveforms
            wav = waveforms[step % len(waveforms)].to(self.device)

            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_features, _ = teacher_model.extract_features(wav)
                teacher_selected = torch.stack(
                    teacher_features[layer_start:layer_end], dim=0
                )
                teacher_emb = teacher_selected.mean(dim=0)  # (1, T, D)

            # Student forward
            student_features, _ = student_model.extract_features(wav)
            student_selected = torch.stack(
                student_features[layer_start:layer_end], dim=0
            )
            student_emb = student_selected.mean(dim=0)  # (1, T, D)

            # Generate mask
            seq_length = student_emb.shape[1]
            mask = objective.generate_mask(seq_length).unsqueeze(0).to(self.device)

            # Compute SSL loss
            ssl_loss = objective(student_emb, teacher_emb, mask)

            # EWC regularization penalty
            ewc_loss = torch.tensor(0.0, device=self.device)
            if (
                baseline_params is not None
                and fisher_diag is not None
                and self.ewc_lambda > 0
            ):
                for name, param in student_model.named_parameters():
                    if name in baseline_params and name in fisher_diag:
                        baseline = baseline_params[name].to(self.device)
                        fisher = fisher_diag[name].to(self.device)
                        ewc_loss += (fisher * (param - baseline) ** 2).sum()

                ewc_loss = (self.ewc_lambda / 2.0) * ewc_loss

            total_loss = ssl_loss + ewc_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm=1.0
            )

            optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if step % 50 == 0:
                logger.info(
                    "Step %d/%d — SSL loss: %.4f, EWC loss: %.4f, Total: %.4f",
                    step,
                    self.num_steps,
                    ssl_loss.item(),
                    ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0.0,
                    loss_val,
                )

        # Bump version
        parts = current_version.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            new_version = f"{parts[0]}_v{int(parts[1]) + 1}"
        else:
            new_version = f"{current_version}_v1"

        return TrainingResult(
            candidate_state_dict=student_model.cpu().state_dict(),
            final_loss=loss_history[-1] if loss_history else float("inf"),
            loss_history=loss_history,
            num_steps=len(loss_history),
            version=new_version,
        )
