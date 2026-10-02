"""
Speech-LLM Multimodal Projector — Connecting Speech Latents to LLM Token Space.

This module implements the Multimodal Audio-Language Projector (Paradigm B)
that maps continuous SSL speech latent embeddings (WavLM/HuBERT 768-d sequence)
directly into the high-dimensional token embedding space of Large Language Models.

Architecture:
    Speech Latent Sequence (T, 768) + Prosodic Features
                     │
                     ▼
       Linear Projection (768 -> 2048 / 4096)
                     │
                     ▼
           GELU Activation + LayerNorm
                     │
                     ▼
       Second Linear Projection (2048 -> Target LLM Dim)
                     │
                     ▼
       Speech Token Sequence for Multimodal LLM Reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpeechLLMProjector(nn.Module):
    """
    Multimodal Speech-to-LLM Projector.
    
    Maps frame-level or pooled self-supervised speech representations (768-d)
    into a target LLM token embedding space (e.g., 2048 or 4096 dimensions).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 2-layer MLP Projector with Residual LayerNorm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        # Attribute Heads (Vocal Emotion, Delivery Tempo, Hesitation Score)
        self.emotion_head = nn.Linear(output_dim, 5)  # Calm, Happy, Angry, Anxious, Neutral
        self.hesitation_head = nn.Linear(output_dim, 1) # Hesitation probability [0, 1]

    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            speech_embeddings: Tensor of shape (Batch, T, 768) or (Batch, 768).

        Returns:
            Projected LLM token embeddings of shape (Batch, T, output_dim) or (Batch, output_dim).
        """
        x = self.fc1(speech_embeddings)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

    def predict_attributes(self, speech_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict acoustic and paralinguistic attributes from projected embeddings."""
        projected = self.forward(speech_embeddings)
        if projected.dim() == 3:
            # Pool across temporal dimension if sequence
            pooled = projected.mean(dim=1)
        else:
            pooled = projected

        emotion_logits = self.emotion_head(pooled)
        hesitation_score = torch.sigmoid(self.hesitation_head(pooled))

        return {
            "projected_embeddings": projected,
            "emotion_logits": emotion_logits,
            "hesitation_score": hesitation_score.squeeze(-1),
        }


def create_multimodal_speech_prompt(
    speech_vector: np.ndarray,
    prosody_stats: Optional[Dict[str, float]] = None,
    query_prompt: str = "Analyze the vocal tone, hesitation, and intent of the speaker."
) -> Dict[str, Any]:
    """
    Utility function to format speech vector representations into a rich
    multimodal context payload for LLM consumption.
    """
    prosody_info = prosody_stats or {}
    
    context = {
        "modalities": ["speech_audio_latent", "text_prompt"],
        "speech_vector_shape": list(speech_vector.shape),
        "mean_magnitude": float(np.linalg.norm(speech_vector)),
        "prosodic_features": {
            "pitch_f0_hz": prosody_info.get("f0_mean", 0.0),
            "speaking_rate_wpm": prosody_info.get("speaking_rate", 0.0),
            "pause_ratio": prosody_info.get("pause_ratio", 0.0),
            "energy_rms": prosody_info.get("rms_mean", 0.0),
        },
        "system_instruction": (
            "You are a Speech Intelligence AI reasoning directly over acoustic, "
            "prosodic, and phonetic latent representations extracted from raw 16kHz audio. "
            "Do NOT treat this as flat text. Evaluate the vocal inflection, delivery style, "
            "and subtle hesitation."
        ),
        "user_query": query_prompt
    }
    return context
