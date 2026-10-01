"""
Foundation Speech Model Embedder Module.

Leverages existing pre-trained speech foundation models (WavLM, HuBERT, Wav2Vec 2.0)
to extract layer-wise, frame-level, and pooled speech representations.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torchaudio
import numpy as np


class FoundationModelEmbedder:
    """
    Interfaces with pre-trained PyTorch/TorchAudio foundation models.
    Supports layer-wise extraction, energy-weighted pooling, and representation disentanglement.
    """

    SUPPORTED_MODELS = {
        "wavlm": torchaudio.pipelines.WAVLM_BASE,
        "hubert": torchaudio.pipelines.HUBERT_BASE,
        "wav2vec2": torchaudio.pipelines.WAV2VEC2_BASE,
    }

    def __init__(self, model_type: str = "wavlm", device: Optional[str] = None):
        model_type = model_type.lower()
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}. Choose from {list(self.SUPPORTED_MODELS.keys())}")

        self.model_type = model_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        bundle = self.SUPPORTED_MODELS[model_type]
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(self.device)
        self.model.eval()

    def extract_layers(self, waveform: np.ndarray, sr: int = 16000) -> Tuple[List[torch.Tensor], np.ndarray]:
        """
        Extract all 12 transformer layer representations from the raw waveform.
        
        Returns:
            layers: List of 12 tensors, each of shape (1, num_frames, 768)
            frame_times: numpy array of timestamp for each frame (seconds)
        """
        # Convert to tensor and ensure 16kHz
        if isinstance(waveform, np.ndarray):
            tensor_wav = torch.from_numpy(waveform).float()
        else:
            tensor_wav = waveform.float()

        if tensor_wav.ndim == 1:
            tensor_wav = tensor_wav.unsqueeze(0)
        elif tensor_wav.ndim == 2 and tensor_wav.shape[0] > 1 and tensor_wav.shape[1] > 1:
            # Multi-channel -> downmix to mono
            tensor_wav = torch.mean(tensor_wav, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            tensor_wav = resampler(tensor_wav)

        tensor_wav = tensor_wav.to(self.device)

        with torch.no_grad():
            # extract_features returns list of 12 layer activations
            layers, _ = self.model.extract_features(tensor_wav)

        # Compute frame timestamps (each frame spans ~20ms, 49 frames per sec)
        num_frames = layers[0].shape[1]
        duration_sec = tensor_wav.shape[1] / self.sample_rate
        frame_times = np.linspace(0, duration_sec, num_frames)

        return layers, frame_times

    def compute_representation_pack(
        self,
        layers: List[torch.Tensor],
        energy_weights: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Computes pooled foundation vectors and disentangled sub-representations:
        - Acoustic vector (Layers 0-3): Channel, vocal tract, timbre
        - Prosodic / Phonetic vector (Layers 4-8): Stress, rhythm, temporal intonation
        - Semantic / Utterance vector (Layers 9-11): Contextual utterance semantics
        - Unified embedding: 768-dim global mean + variance embedding
        """
        # layers is list of 12 tensors (1, T, 768)
        stacked = torch.stack(layers, dim=0).squeeze(1).cpu()  # (12, T, 768)
        num_layers, T, D = stacked.shape

        # Weighting frames by vocal energy (if provided) to focus on voiced speech
        if energy_weights is not None and len(energy_weights) > 0:
            # Interpolate energy weights to match frame length T
            w = torch.from_numpy(energy_weights).float()
            if len(w) != T:
                w = torch.nn.functional.interpolate(
                    w.view(1, 1, -1), size=T, mode='linear', align_corners=False
                ).view(-1)
            w = torch.clamp(w, min=1e-4)
            w = (w / w.sum()).view(1, T, 1)  # (1, T, 1)
        else:
            w = torch.full((1, T, 1), 1.0 / T)

        # Weighted temporal pooling across each layer
        weighted_layer_means = (stacked * w).sum(dim=1)  # (12, 768)
        layer_stds = stacked.std(dim=1)                  # (12, 768) dynamic variation

        # Disentangled representations
        # Research shows Layers 0-3 contain acoustic timbre, 4-8 contain prosody/phonetics, 9-11 contain semantic context
        acoustic_emb = weighted_layer_means[0:4].mean(dim=0).numpy()
        prosodic_emb = weighted_layer_means[4:9].mean(dim=0).numpy()
        semantic_emb = weighted_layer_means[9:12].mean(dim=0).numpy()
        
        # Unified final embedding (Layer 11 + middle layer prosodic mixture)
        unified_emb = 0.5 * weighted_layer_means[11].numpy() + 0.5 * prosodic_emb

        # Layer-wise norm for visualization
        layer_activity = stacked.norm(dim=2).numpy()  # (12, T)

        return {
            "unified_embedding": unified_emb.astype(np.float32),
            "acoustic_embedding": acoustic_emb.astype(np.float32),
            "prosodic_embedding": prosodic_emb.astype(np.float32),
            "semantic_embedding": semantic_emb.astype(np.float32),
            "layer_activity_matrix": layer_activity.astype(np.float32),
            "temporal_variation": layer_stds.mean(dim=0).numpy().astype(np.float32),
        }
