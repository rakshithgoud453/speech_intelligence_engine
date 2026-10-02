"""
Representation Engine — Subsystem A.

Converts raw audio into temporal latent sequences using pretrained
self-supervised speech foundation models.
"""

from .encoder import SpeechEncoder
from .segmenter import VoiceActivitySegmenter, Segment

__all__ = ["SpeechEncoder", "VoiceActivitySegmenter", "Segment"]
