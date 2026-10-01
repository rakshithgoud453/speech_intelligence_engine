"""
Speech Intelligence Engine: A foundation-model-powered representation layer for speech.

Extracts 'how it was said' (prosody, rhythm, pauses, pitch, emotional coloring)
using pretrained speech foundation models (WavLM, HuBERT, Wav2Vec2) and acoustic profiling.
"""

from .core import SpeechIntelligenceEngine, SpeechIntelligenceResult
from .prosody import ProsodicAnalyzer, ProsodyProfile
from .foundation import FoundationModelEmbedder
from .visualizer import SpeechVisualizer

__all__ = [
    "SpeechIntelligenceEngine",
    "SpeechIntelligenceResult",
    "ProsodicAnalyzer",
    "ProsodyProfile",
    "FoundationModelEmbedder",
    "SpeechVisualizer",
]
