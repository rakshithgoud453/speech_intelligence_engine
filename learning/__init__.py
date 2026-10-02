"""
Continual Learning Engine — Subsystem D.

Responsible for evolving the encoder over time:
    - SelfSupervisedObjective: Masked prediction loss for speech
    - CandidateTrainer:        Trains a new encoder using replay memory
    - StabilityTester:         Validates candidate before promotion (EWC-inspired)
"""

from .objectives import SelfSupervisedObjective
from .trainer import CandidateTrainer, TrainingResult
from .stability import StabilityTester, StabilityReport

__all__ = [
    "SelfSupervisedObjective",
    "CandidateTrainer",
    "TrainingResult",
    "StabilityTester",
    "StabilityReport",
]
