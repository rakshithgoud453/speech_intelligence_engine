"""
Prosody & Acoustic Feature Extraction Module.

Extracts interpretable, signal-level speech dynamics:
- Pitch (F0) contour & dynamics (inflection, expressiveness, question/statement cadence)
- Energy & Loudness envelope (dynamic range, emphasis bursts)
- Rhythm, Pauses & Speaking Rate (silence ratio, hesitation count, syllable rate)
- Voice quality indicators (spectral centroid, flux, stability)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
import soundfile as sf


@dataclass
class ProsodyProfile:
    duration_sec: float
    speech_duration_sec: float
    silence_duration_sec: float
    pause_ratio: float                # % of audio that is pause/silence
    num_pauses: int                   # number of pauses (>180ms)
    mean_pause_duration_sec: float    # average pause length
    
    # Pitch (F0) dynamics
    mean_pitch_hz: float
    std_pitch_hz: float
    median_pitch_hz: float
    pitch_range_semitones: float      # octave/semitone pitch spread (expressiveness)
    terminal_pitch_slope: float       # rising (+, questioning/hesitant) vs falling (-, decisive)
    pitch_variability_label: str      # 'monotone', 'moderate', 'expressive', 'highly_dynamic'
    
    # Rhythm & Energy dynamics
    estimated_speaking_rate_syllables_sec: float  # syllables per second of voiced speech
    speech_tempo_label: str           # 'very_slow', 'deliberate', 'moderate', 'rapid'
    rms_energy_mean: float
    dynamic_range_db: float           # dynamic volume variation
    emphasis_peaks_per_sec: float     # vocal emphasis spikes
    
    # Voice Quality & Timbral brightness
    spectral_centroid_hz: float       # acoustic brightness / vocal resonance
    spectral_flux_mean: float         # rate of spectral change (articulatory speed)
    zero_crossing_rate_mean: float    # breathiness / fricative presence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProsodicAnalyzer:
    """
    Analyzes raw audio waveform to quantify the paralinguistic 'how it was said' metrics.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze(self, y: np.ndarray, sr: int = None) -> Tuple[ProsodyProfile, Dict[str, np.ndarray]]:
        """
        Analyze audio array and return structured ProsodyProfile and time-series arrays.
        """
        if sr is None:
            sr = self.sample_rate
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Normalize waveform
        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        duration_sec = len(y) / sr
        frame_length = 512              # 32ms window (ensures >= 2 periods of 65Hz)
        hop_length = int(sr * 0.010)    # 10ms step (160 samples)

        # 1. Energy & RMS
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # 2. Voice Activity & Pause Analysis
        # Dynamic threshold based on percentile
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        speech_thresh_db = -35.0  # frames above -35 dB considered speech
        is_speech = rms_db > speech_thresh_db

        # Detect pauses: contiguous blocks of silence > 180ms
        min_pause_frames = int(0.180 / (hop_length / sr))
        pauses = []
        in_pause = False
        pause_start = 0

        for i, speech in enumerate(is_speech):
            if not speech and not in_pause:
                in_pause = True
                pause_start = i
            elif speech and in_pause:
                in_pause = False
                pause_len = i - pause_start
                if pause_len >= min_pause_frames:
                    pauses.append((pause_start * hop_length / sr, i * hop_length / sr))

        if in_pause and (len(is_speech) - pause_start) >= min_pause_frames:
            pauses.append((pause_start * hop_length / sr, len(is_speech) * hop_length / sr))

        total_pause_sec = sum(p[1] - p[0] for p in pauses)
        pause_ratio = (total_pause_sec / duration_sec) if duration_sec > 0 else 0.0
        num_pauses = len(pauses)
        mean_pause_sec = (total_pause_sec / num_pauses) if num_pauses > 0 else 0.0
        speech_duration_sec = max(0.0, duration_sec - total_pause_sec)

        # 3. Fundamental Frequency (F0 / Pitch) Estimation
        # Fast YIN algorithm bounded between 65Hz (C2) and 500Hz (B4/soprano)
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
        # Mask unvoiced / silent frames
        voiced_mask = is_speech[:len(f0)] & (f0 > 65) & (f0 < 490)
        voiced_f0 = f0[voiced_mask]

        if len(voiced_f0) > 5:
            mean_pitch = float(np.mean(voiced_f0))
            std_pitch = float(np.std(voiced_f0))
            median_pitch = float(np.median(voiced_f0))
            p05, p95 = np.percentile(voiced_f0, [5, 95])
            if p05 > 0:
                pitch_range_st = float(12.0 * np.log2(p95 / p05))
            else:
                pitch_range_st = 0.0

            # Terminal pitch slope: last 30% of voiced speech
            cutoff = int(len(voiced_f0) * 0.7)
            terminal_voiced = voiced_f0[cutoff:]
            if len(terminal_voiced) > 3:
                x = np.arange(len(terminal_voiced))
                slope, _ = np.polyfit(x, terminal_voiced, 1)
                terminal_pitch_slope = float(slope)
            else:
                terminal_pitch_slope = 0.0
        else:
            mean_pitch = 0.0
            std_pitch = 0.0
            median_pitch = 0.0
            pitch_range_st = 0.0
            terminal_pitch_slope = 0.0

        # Categorize pitch variability
        if pitch_range_st < 3.0:
            pitch_label = "monotone"
        elif pitch_range_st < 7.0:
            pitch_label = "moderate"
        elif pitch_range_st < 12.0:
            pitch_label = "expressive"
        else:
            pitch_label = "highly_dynamic"

        # 4. Syllable Rate / Rhythm Estimation
        # Syllable nuclei detected via local RMS energy peaks in voiced regions
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        num_syllables = len(peaks)
        speaking_rate = (num_syllables / speech_duration_sec) if speech_duration_sec > 0.5 else 0.0

        if speaking_rate < 2.5:
            tempo_label = "very_slow"
        elif speaking_rate < 3.8:
            tempo_label = "deliberate"
        elif speaking_rate < 5.2:
            tempo_label = "moderate"
        else:
            tempo_label = "rapid"

        # 5. Dynamic Range & Energy Peak Dynamics
        voiced_rms = rms[is_speech[:len(rms)]]
        if len(voiced_rms) > 0:
            dyn_range = float(np.max(librosa.amplitude_to_db(voiced_rms)) - np.min(librosa.amplitude_to_db(voiced_rms)))
        else:
            dyn_range = 0.0

        # Emphasis peaks (bursts > 1.5 standard deviations above mean)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        emphasis_peaks = np.sum((rms > (rms_mean + 1.5 * rms_std)))
        emphasis_rate = float(emphasis_peaks / duration_sec) if duration_sec > 0 else 0.0

        # 6. Spectral & Voice Quality
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid_mean = float(np.mean(cent[is_speech[:len(cent)]])) if np.any(is_speech[:len(cent)]) else 0.0

        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        flux_mean = float(np.mean(flux))

        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr))

        profile = ProsodyProfile(
            duration_sec=round(duration_sec, 3),
            speech_duration_sec=round(speech_duration_sec, 3),
            silence_duration_sec=round(total_pause_sec, 3),
            pause_ratio=round(pause_ratio, 3),
            num_pauses=num_pauses,
            mean_pause_duration_sec=round(mean_pause_sec, 3),
            mean_pitch_hz=round(mean_pitch, 2),
            std_pitch_hz=round(std_pitch, 2),
            median_pitch_hz=round(median_pitch, 2),
            pitch_range_semitones=round(pitch_range_st, 2),
            terminal_pitch_slope=round(terminal_pitch_slope, 3),
            pitch_variability_label=pitch_label,
            estimated_speaking_rate_syllables_sec=round(speaking_rate, 2),
            speech_tempo_label=tempo_label,
            rms_energy_mean=round(rms_mean, 4),
            dynamic_range_db=round(dyn_range, 2),
            emphasis_peaks_per_sec=round(emphasis_rate, 2),
            spectral_centroid_hz=round(centroid_mean, 1),
            spectral_flux_mean=round(flux_mean, 3),
            zero_crossing_rate_mean=round(zcr_mean, 4),
        )

        time_series = {
            "times": times,
            "rms": rms,
            "f0": f0,
            "voiced_mask": voiced_mask,
            "pauses": np.array(pauses),
            "waveform": y,
        }

        return profile, time_series
