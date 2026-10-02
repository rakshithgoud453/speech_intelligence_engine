"""
Self-Supervised Objectives for Speech.

Defines the training objective used to evolve the encoder.
The system uses a masked prediction objective inspired by HuBERT:
mask portions of the input audio and train the encoder to predict
the missing representations from context.

This is NOT a new invention — it is the same masked prediction
paradigm used by HuBERT, wav2vec 2.0, and data2vec, adapted
for continual fine-tuning of an already-pretrained model.

The key difference from standard pretraining:
    - We train on a SMALL amount of data (replay buffer contents)
    - We use EWC regularization to prevent catastrophic forgetting
    - The target representations come from the CURRENT encoder
      (teacher-student, where teacher = current θ, student = candidate θ')
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfSupervisedObjective(nn.Module):
    """
    Masked frame prediction objective for continual encoder evolution.

    Given a waveform, the objective:
        1. Encodes it with the CURRENT (frozen) teacher encoder → target frames.
        2. Masks a random subset of frames in the student encoder's input.
        3. Trains the student to predict the teacher's representations
           at the masked positions.

    This is a simplified version of the data2vec / HuBERT objective,
    designed for fine-tuning rather than pretraining from scratch.

    Parameters:
        mask_prob: Probability of masking each frame. Default 0.15 (15%).
        mask_length: Number of consecutive frames to mask per span. Default 10.
        embedding_dim: Dimension of the encoder output (768 for base models).
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_length: int = 10,
        embedding_dim: int = 768,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.embedding_dim = embedding_dim

        # Projection head: maps student representations to prediction space
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def generate_mask(self, seq_length: int) -> torch.Tensor:
        """
        Generate a span-based mask for the temporal sequence.

        Returns:
            Boolean tensor of shape (seq_length,) where True = masked.
        """
        mask = torch.zeros(seq_length, dtype=torch.bool)
        num_masks = max(1, int(seq_length * self.mask_prob / self.mask_length))

        for _ in range(num_masks):
            start = torch.randint(0, max(1, seq_length - self.mask_length), (1,)).item()
            end = min(start + self.mask_length, seq_length)
            mask[start:end] = True

        return mask

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the masked prediction loss.

        Args:
            student_features: Student encoder output, shape (B, T, D).
            teacher_features: Teacher (frozen) encoder output, shape (B, T, D).
                              These are the prediction targets.
            mask: Boolean mask, shape (B, T). True = masked positions.

        Returns:
            Scalar loss (mean smooth L1 over masked positions).
        """
        # Project student features
        predicted = self.projection(student_features)  # (B, T, D)

        # Extract masked positions
        # mask: (B, T) → expand to (B, T, D)
        mask_expanded = mask.unsqueeze(-1).expand_as(predicted)

        pred_masked = predicted[mask_expanded].view(-1, self.embedding_dim)
        target_masked = teacher_features[mask_expanded.detach()].view(
            -1, self.embedding_dim
        )

        if pred_masked.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Smooth L1 loss (Huber loss) — more robust than MSE for high-dim targets
        loss = F.smooth_l1_loss(pred_masked, target_masked.detach())

        return loss
