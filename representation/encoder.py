"""
Speech Encoder — Foundation Model Wrapper.

Wraps pretrained self-supervised speech models (HuBERT, WavLM, Wav2Vec 2.0)
from torchaudio to extract layer-wise temporal representations.

This module is the "sensory cortex" of the learning engine.
It does NOT pool or collapse the temporal dimension — the output
is always a sequence (T, D) preserving the temporal structure of speech.

The encoder can be swapped or fine-tuned by the Continual Learning Engine
without affecting the rest of the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class TemporalEmbedding:
    """
    The output of encoding a speech segment.

    This is the fundamental unit of experience in the learning system.
    It preserves the full temporal sequence — it is NOT a single vector.

    Attributes:
        sequence: Temporal embedding matrix of shape (T, D).
                  T = number of time frames (~50 per second of audio).
                  D = embedding dimension (768 for base models).
        frame_times: Timestamps in seconds for each frame, shape (T,).
        duration_sec: Total duration of the source audio in seconds.
        model_version: Identifier of the encoder that produced this embedding.
    """

    sequence: np.ndarray        # (T, D) — the temporal latent sequence
    frame_times: np.ndarray     # (T,)   — timestamp per frame
    duration_sec: float
    model_version: str

    @property
    def num_frames(self) -> int:
        return self.sequence.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.sequence.shape[1]

    def pooled(self, energy_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Produce a single (D,) vector via energy-weighted temporal pooling.
        Used for nearest-neighbor search in prototype memory.

        Args:
            energy_weights: Optional per-frame energy weights of shape (T,).
                            If None, uses uniform mean pooling.

        Returns:
            Pooled vector of shape (D,).
        """
        if energy_weights is not None and len(energy_weights) > 0:
            # Interpolate weights to match frame count if needed
            if len(energy_weights) != self.num_frames:
                w = np.interp(
                    np.linspace(0, 1, self.num_frames),
                    np.linspace(0, 1, len(energy_weights)),
                    energy_weights,
                )
            else:
                w = energy_weights.copy()

            w = np.clip(w, 1e-6, None)
            w = w / w.sum()
            return (self.sequence * w[:, np.newaxis]).sum(axis=0).astype(np.float32)

        return self.sequence.mean(axis=0).astype(np.float32)


# ─── Model registry ───────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, torchaudio.pipelines.Wav2Vec2Bundle] = {
    "wavlm": torchaudio.pipelines.WAVLM_BASE,
    "hubert": torchaudio.pipelines.HUBERT_BASE,
    "wav2vec2": torchaudio.pipelines.WAV2VEC2_BASE,
}

SUPPORTED_MODELS = list(_MODEL_REGISTRY.keys())


class SpeechEncoder:
    """
    Wraps a pretrained speech foundation model for temporal embedding extraction.

    The encoder exposes two levels of output:
        1. Full layer-wise extraction (12 transformer layers)
        2. Configurable layer-range extraction for disentangled representations

    This class is designed to be swappable — the Continual Learning Engine
    can replace the underlying model weights while keeping the same interface.

    Parameters:
        model_type: One of 'wavlm', 'hubert', 'wav2vec2'.
        device: PyTorch device string. Auto-detected if None.
        extraction_layers: Which transformer layers to average for the
                           primary temporal embedding. Default is layers 4-8
                           (prosodic/phonetic range per SSL literature).
    """

    def __init__(
        self,
        model_type: str = "hubert",
        device: Optional[str] = None,
        extraction_layers: Optional[Tuple[int, int]] = None,
    ):
        model_type = model_type.lower()
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model: {model_type}. "
                f"Choose from {SUPPORTED_MODELS}"
            )

        self.model_type = model_type

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        bundle = _MODEL_REGISTRY[model_type]
        self.sample_rate: int = bundle.sample_rate
        self._model = bundle.get_model().to(self.device)
        self._model.eval()

        # Default: layers 4-8 capture prosodic/phonetic structure
        self.extraction_layers = extraction_layers or (4, 9)

        # Version string for tracking which encoder produced an embedding
        self._version = f"{model_type}_base_v0"

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    def version(self, v: str) -> None:
        self._version = v

    def encode(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
    ) -> TemporalEmbedding:
        """
        Encode a waveform into a temporal latent sequence.

        This is the primary interface used by the learning engine.
        The output preserves the full temporal dimension.

        Args:
            waveform: 1-D float32 audio array.
            sr: Sample rate of the input audio.

        Returns:
            TemporalEmbedding containing the (T, D) sequence.
        """
        tensor_wav = self._prepare_waveform(waveform, sr)

        with torch.no_grad():
            all_layers, _ = self._model.extract_features(tensor_wav)

        # Average over the selected layer range
        layer_start, layer_end = self.extraction_layers
        selected = torch.stack(all_layers[layer_start:layer_end], dim=0)
        # selected: (num_layers, 1, T, D)
        averaged = selected.mean(dim=0).squeeze(0).cpu().numpy()
        # averaged: (T, D)

        num_frames = averaged.shape[0]
        duration_sec = tensor_wav.shape[1] / self.sample_rate
        frame_times = np.linspace(0, duration_sec, num_frames)

        return TemporalEmbedding(
            sequence=averaged.astype(np.float32),
            frame_times=frame_times,
            duration_sec=duration_sec,
            model_version=self._version,
        )

    def encode_all_layers(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
    ) -> List[np.ndarray]:
        """
        Extract all 12 transformer layer activations.

        Returns:
            List of 12 arrays, each of shape (T, D).
        """
        tensor_wav = self._prepare_waveform(waveform, sr)

        with torch.no_grad():
            all_layers, _ = self._model.extract_features(tensor_wav)

        return [
            layer.squeeze(0).cpu().numpy().astype(np.float32)
            for layer in all_layers
        ]

    def get_model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model for fine-tuning."""
        return self._model

    def load_weights(self, state_dict: dict) -> None:
        """Load new weights into the encoder (used by Continual Learning)."""
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def _prepare_waveform(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        """Convert numpy waveform to a properly formatted tensor."""
        if isinstance(waveform, np.ndarray):
            tensor_wav = torch.from_numpy(waveform).float()
        else:
            tensor_wav = waveform.float()

        if tensor_wav.ndim == 1:
            tensor_wav = tensor_wav.unsqueeze(0)
        elif tensor_wav.ndim == 2 and tensor_wav.shape[0] > 1:
            # Multi-channel → downmix to mono
            tensor_wav = tensor_wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            tensor_wav = resampler(tensor_wav)

        # Pad short waveforms to minimum 1600 samples (0.1s) to prevent zero-frame SSL outputs
        if tensor_wav.shape[1] < 1600:
            tensor_wav = torch.nn.functional.pad(tensor_wav, (0, 1600 - tensor_wav.shape[1]))

        return tensor_wav.to(self.device)

