"""
Representation Engine — Subsystem A.

Converts raw audio into temporal latent sequences using pretrained
self-supervised speech foundation models.
"""

from .encoder import SpeechEncoder
from .segmenter import VoiceActivitySegmenter, Segment
from .projector import SpeechLLMProjector, create_multimodal_speech_prompt

__all__ = [
    "SpeechEncoder",
    "VoiceActivitySegmenter",
    "Segment",
    "SpeechLLMProjector",
    "create_multimodal_speech_prompt",
]

