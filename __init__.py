"""
Speech Intelligence Engine: A foundation-model-powered representation layer for speech.

Extracts 'how it was said' (prosody, rhythm, pauses, pitch, emotional coloring)
using pretrained speech foundation models (WavLM, HuBERT, Wav2Vec2) and acoustic profiling.
"""

from .engine import SpeechLearningEngine, AssimilationResult, EngineState
from .prosody import ProsodicAnalyzer, ProsodyProfile

__all__ = [
    "SpeechLearningEngine",
    "AssimilationResult",
    "EngineState",
    "ProsodicAnalyzer",
    "ProsodyProfile",
]

