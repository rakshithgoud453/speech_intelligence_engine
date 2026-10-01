"""
Core Speech Intelligence Engine.

Orchestrates pretrained speech foundation models with signal-level prosodic profiling,
speech delivery comparison, and LLM context synthesis.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union, Tuple, List
import os
import json
import numpy as np
import librosa
import soundfile as sf

from .prosody import ProsodicAnalyzer, ProsodyProfile
from .foundation import FoundationModelEmbedder


@dataclass
class SpeechIntelligenceResult:
    audio_path: Optional[str]
    duration_sec: float
    delivery_style_summary: str
    dominant_cues: List[str]
    prosody_profile: ProsodyProfile
    embeddings: Dict[str, np.ndarray]
    llm_context_text: str
    llm_context_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "duration_sec": self.duration_sec,
            "delivery_style_summary": self.delivery_style_summary,
            "dominant_cues": self.dominant_cues,
            "prosody_profile": self.prosody_profile.to_dict(),
            "llm_context_json": self.llm_context_json,
            "llm_context_text": self.llm_context_text,
            "embedding_shapes": {k: list(v.shape) for k, v in self.embeddings.items() if isinstance(v, np.ndarray)},
        }


class SpeechIntelligenceEngine:
    """
    Main entry point for extracting deep speech intelligence representations.
    Combines self-supervised foundation models with explicit paralinguistic analysis.
    """

    def __init__(self, model_type: str = "wavlm", device: Optional[str] = None):
        self.model_type = model_type
        self.prosody_analyzer = ProsodicAnalyzer(sample_rate=16000)
        self.foundation_embedder = FoundationModelEmbedder(model_type=model_type, device=device)
        self.sample_rate = 16000

    def load_audio(self, audio_input: Union[str, np.ndarray], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file or validate array, ensuring 16kHz mono.
        """
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            y, file_sr = librosa.load(audio_input, sr=self.sample_rate, mono=True)
            return y, self.sample_rate
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
            if y.ndim > 1:
                y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
            if sr is not None and sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            return y.astype(np.float32), self.sample_rate
        else:
            raise TypeError("audio_input must be a file path string or numpy ndarray")

    def infer_delivery_style(self, p: ProsodyProfile) -> Tuple[str, List[str]]:
        """
        Synthesize signal metrics into an interpretable delivery style label and key cues.
        """
        cues = []
        tags = []

        # Pause and hesitation cues
        if p.pause_ratio > 0.35:
            tags.append("Hesitant / Contemplative")
            cues.append(f"High pause ratio ({int(p.pause_ratio * 100)}% silence), indicating hesitation or deliberation.")
        elif p.pause_ratio < 0.12 and p.duration_sec > 1.5:
            tags.append("Fluent / Rapid-Fire")
            cues.append("Very low pause ratio (<12%), speech flows continuously without hesitation.")

        # Tempo cues
        if p.speech_tempo_label in ["very_slow", "deliberate"]:
            tags.append("Deliberate Tempo")
            cues.append(f"Deliberate speaking rate ({p.estimated_speaking_rate_syllables_sec} syl/sec).")
        elif p.speech_tempo_label == "rapid":
            tags.append("Urgent / Fast-Paced")
            cues.append(f"Fast speaking tempo ({p.estimated_speaking_rate_syllables_sec} syl/sec).")

        # Pitch dynamics cues
        if p.terminal_pitch_slope > 1.5:
            tags.append("Rising Cadence (Questioning / Uncertain)")
            cues.append(f"Distinct rising terminal pitch contour (+{p.terminal_pitch_slope:.1f}), characteristic of a question or doubt.")
        elif p.terminal_pitch_slope < -1.5:
            tags.append("Decisive Cadence (Assertive / Firm)")
            cues.append(f"Falling terminal pitch contour ({p.terminal_pitch_slope:.1f}), projecting decisiveness and finality.")

        if p.pitch_variability_label == "monotone":
            tags.append("Monotone / Flat")
            cues.append(f"Compressed pitch range ({p.pitch_range_semitones} semitones), flat emotional inflection.")
        elif p.pitch_variability_label in ["expressive", "highly_dynamic"]:
            tags.append("Expressive / Animated")
            cues.append(f"Expansive pitch variation ({p.pitch_range_semitones} semitones), energetic intonation.")

        # Energy dynamics
        if p.dynamic_range_db > 28.0:
            cues.append(f"High dynamic energy range ({p.dynamic_range_db:.1f} dB) with strong vocal stress accents.")
        elif p.dynamic_range_db < 15.0 and p.duration_sec > 1.0:
            cues.append(f"Subdued dynamic energy ({p.dynamic_range_db:.1f} dB), restrained vocal projection.")

        if not tags:
            tags = ["Calm / Conversational"]
            cues.append("Balanced tempo and natural prosodic inflection.")

        summary_style = " | ".join(tags)
        return summary_style, cues

    def generate_llm_context(
        self,
        audio_name: str,
        style_summary: str,
        cues: List[str],
        profile: ProsodyProfile,
        transcript: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build an LLM-ready prompt section and JSON payload encoding how the speech was spoken.
        """
        json_payload = {
            "speaker_audio": audio_name,
            "transcript": transcript or "[No text transcript provided]",
            "delivery_style": style_summary,
            "paralinguistic_cues": cues,
            "metrics": {
                "speaking_rate_syl_sec": profile.estimated_speaking_rate_syllables_sec,
                "pause_ratio_percent": round(profile.pause_ratio * 100, 1),
                "num_pauses": profile.num_pauses,
                "mean_pause_sec": profile.mean_pause_duration_sec,
                "pitch_range_semitones": profile.pitch_range_semitones,
                "pitch_contour_trend": "Rising" if profile.terminal_pitch_slope > 0.5 else ("Falling" if profile.terminal_pitch_slope < -0.5 else "Neutral"),
                "pitch_variability": profile.pitch_variability_label,
                "dynamic_range_db": profile.dynamic_range_db,
            },
        }

        cue_bullets = "\n".join([f"- {c}" for c in cues])
        text_prompt = f"""[SPEECH INTELLIGENCE & PROSODIC CONTEXT]
Input Audio: {audio_name}
{f'Words Spoken: "{transcript}"' if transcript else ''}
Delivery Style: {style_summary}
Paralinguistic Breakdown:
{cue_bullets}
Acoustic Metrics:
- Tempo: {profile.speech_tempo_label} ({profile.estimated_speaking_rate_syllables_sec} syllables/sec)
- Hesitation/Pauses: {profile.num_pauses} pauses ({round(profile.pause_ratio * 100, 1)}% of duration)
- Pitch Inflection: {profile.pitch_variability_label} ({profile.pitch_range_semitones} st range, {'Rising' if profile.terminal_pitch_slope > 0.5 else 'Falling' if profile.terminal_pitch_slope < -0.5 else 'Flat'} terminal slope)
- Energy Dynamic Range: {profile.dynamic_range_db} dB

Instructions for Reasoning Model:
Do NOT treat the spoken words as flat text. Interpret the speaker's emotional state, intent, and conviction in accordance with the above acoustic and prosodic evidence."""

        return text_prompt, json_payload

    def analyze(
        self,
        audio_input: Union[str, np.ndarray],
        transcript: Optional[str] = None
    ) -> SpeechIntelligenceResult:
        """
        Analyze audio and produce full deep speech intelligence representation.
        """
        audio_name = os.path.basename(audio_input) if isinstance(audio_input, str) else "in_memory_audio"
        waveform, sr = self.load_audio(audio_input)

        # 1. Signal-level prosodic profiling
        profile, time_series = self.prosody_analyzer.analyze(waveform, sr=sr)

        # 2. Deep foundation model layer extraction
        layers, frame_times = self.foundation_embedder.extract_layers(waveform, sr=sr)
        rep_pack = self.foundation_embedder.compute_representation_pack(
            layers=layers,
            energy_weights=time_series["rms"]
        )

        # 3. Infer delivery style and cues
        style_summary, cues = self.infer_delivery_style(profile)

        # 4. Generate LLM prompt injection & structured payload
        llm_text, llm_json = self.generate_llm_context(
            audio_name=audio_name,
            style_summary=style_summary,
            cues=cues,
            profile=profile,
            transcript=transcript
        )

        return SpeechIntelligenceResult(
            audio_path=audio_input if isinstance(audio_input, str) else None,
            duration_sec=profile.duration_sec,
            delivery_style_summary=style_summary,
            dominant_cues=cues,
            prosody_profile=profile,
            embeddings=rep_pack,
            llm_context_text=llm_text,
            llm_context_json=llm_json,
        )

    def compare(
        self,
        audio1: Union[str, np.ndarray],
        audio2: Union[str, np.ndarray],
        transcript1: Optional[str] = None,
        transcript2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two speech clips across foundation embeddings and prosodic dimensions.
        Determines how similar the *delivery* is, independent of the text.
        """
        res1 = self.analyze(audio1, transcript=transcript1)
        res2 = self.analyze(audio2, transcript=transcript2)

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            return float(dot / norm)

        # Foundation representation similarities
        global_sim = cosine_sim(res1.embeddings["unified_embedding"], res2.embeddings["unified_embedding"])
        prosodic_rep_sim = cosine_sim(res1.embeddings["prosodic_embedding"], res2.embeddings["prosodic_embedding"])
        acoustic_rep_sim = cosine_sim(res1.embeddings["acoustic_embedding"], res2.embeddings["acoustic_embedding"])

        # Prosodic metric distances
        p1 = res1.prosody_profile
        p2 = res2.prosody_profile

        tempo_delta = abs(p1.estimated_speaking_rate_syllables_sec - p2.estimated_speaking_rate_syllables_sec)
        pause_ratio_delta = abs(p1.pause_ratio - p2.pause_ratio)
        pitch_range_delta = abs(p1.pitch_range_semitones - p2.pitch_range_semitones)
        pitch_slope_delta = abs(p1.terminal_pitch_slope - p2.terminal_pitch_slope)

        # Style matching assessment
        if prosodic_rep_sim > 0.88 and pause_ratio_delta < 0.15:
            delivery_match = "High Delivery Concordance: Both speakers share very similar tempo, rhythm, and vocal energy."
        elif prosodic_rep_sim > 0.75:
            delivery_match = "Moderate Delivery Concordance: Comparable overall cadence with noticeable nuance differences."
        else:
            delivery_match = "Divergent Delivery Style: Distinct rhythm, hesitation, or intonation patterns."

        return {
            "clip_1": {
                "name": res1.audio_path or "Clip 1",
                "style": res1.delivery_style_summary,
                "speaking_rate": p1.estimated_speaking_rate_syllables_sec,
                "pause_ratio": round(p1.pause_ratio * 100, 1),
                "pitch_range_st": p1.pitch_range_semitones,
            },
            "clip_2": {
                "name": res2.audio_path or "Clip 2",
                "style": res2.delivery_style_summary,
                "speaking_rate": p2.estimated_speaking_rate_syllables_sec,
                "pause_ratio": round(p2.pause_ratio * 100, 1),
                "pitch_range_st": p2.pitch_range_semitones,
            },
            "similarity_metrics": {
                "foundation_unified_cosine_similarity": round(global_sim, 4),
                "foundation_prosodic_layer_similarity": round(prosodic_rep_sim, 4),
                "foundation_acoustic_timbre_similarity": round(acoustic_rep_sim, 4),
                "tempo_difference_syl_sec": round(tempo_delta, 2),
                "pause_ratio_difference_percent": round(pause_ratio_delta * 100, 1),
                "pitch_range_difference_semitones": round(pitch_range_delta, 2),
            },
            "verdict": delivery_match,
        }
