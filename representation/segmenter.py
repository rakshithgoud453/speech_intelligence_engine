"""
Voice Activity Segmenter.

Splits continuous audio into voiced segments using energy-based
voice activity detection (VAD). This is the first stage of the pipeline:
raw audio is broken into meaningful speech episodes before encoding.

Algorithm:
    - Compute frame-level RMS energy
    - Apply adaptive dB threshold to classify frames as speech/silence
    - Merge adjacent speech frames into contiguous segments
    - Filter out segments shorter than min_duration_ms
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa


@dataclass(frozen=True)
class Segment:
    """An immutable speech segment within a recording."""

    start_ms: int
    end_ms: int
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


class VoiceActivitySegmenter:
    """
    Energy-based voice activity detector and segmenter.

    Splits a continuous audio stream into discrete speech segments,
    discarding silence and non-speech regions. Each segment becomes
    an independent "experience" for the learning engine.

    Parameters:
        sample_rate: Target sample rate (default 16000 Hz).
        frame_length: Analysis window size in samples.
        hop_length: Step size between frames in samples.
        silence_threshold_db: dB below peak to classify as silence.
        min_segment_ms: Minimum segment duration to keep (ms).
        min_silence_ms: Minimum silence duration to trigger a split (ms).
        merge_gap_ms: Merge segments separated by less than this gap (ms).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 160,
        silence_threshold_db: float = -35.0,
        min_segment_ms: int = 300,
        min_silence_ms: int = 200,
        merge_gap_ms: int = 100,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.silence_threshold_db = silence_threshold_db
        self.min_segment_ms = min_segment_ms
        self.min_silence_ms = min_silence_ms
        self.merge_gap_ms = merge_gap_ms

    def segment(self, waveform: np.ndarray, sr: int | None = None) -> List[Segment]:
        """
        Segment a waveform into voiced speech regions.

        Args:
            waveform: 1-D float32 audio array.
            sr: Sample rate of the input. Resampled to self.sample_rate if different.

        Returns:
            List of Segment objects, each containing an isolated speech region.
        """
        if sr is not None and sr != self.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sample_rate
            )

        waveform = waveform.astype(np.float32)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        # Compute frame-level RMS energy
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Classify frames as speech or silence
        is_speech = rms_db > self.silence_threshold_db

        # Extract contiguous speech regions as (start_frame, end_frame) pairs
        raw_regions = self._extract_regions(is_speech)

        # Convert frame indices to milliseconds
        ms_regions = [
            (
                int(start * self.hop_length * 1000 / self.sample_rate),
                int(end * self.hop_length * 1000 / self.sample_rate),
            )
            for start, end in raw_regions
        ]

        # Merge regions that are very close together
        merged = self._merge_close_regions(ms_regions, self.merge_gap_ms)

        # Filter out short segments and silence gaps
        min_silence_frames = int(
            self.min_silence_ms * self.sample_rate / (1000 * self.hop_length)
        )
        segments: List[Segment] = []
        for start_ms, end_ms in merged:
            duration_ms = end_ms - start_ms
            if duration_ms < self.min_segment_ms:
                continue

            # Extract the waveform slice
            start_sample = int(start_ms * self.sample_rate / 1000)
            end_sample = int(end_ms * self.sample_rate / 1000)
            end_sample = min(end_sample, len(waveform))

            segment_wav = waveform[start_sample:end_sample].copy()
            if len(segment_wav) == 0:
                continue

            segments.append(
                Segment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    waveform=segment_wav,
                    sample_rate=self.sample_rate,
                )
            )

        return segments

    def _extract_regions(
        self, is_speech: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Extract contiguous True regions as (start, end) frame index pairs."""
        regions: List[Tuple[int, int]] = []
        in_region = False
        start = 0

        for i, active in enumerate(is_speech):
            if active and not in_region:
                in_region = True
                start = i
            elif not active and in_region:
                in_region = False
                regions.append((start, i))

        if in_region:
            regions.append((start, len(is_speech)))

        return regions

    def _merge_close_regions(
        self, regions: List[Tuple[int, int]], gap_ms: int
    ) -> List[Tuple[int, int]]:
        """Merge regions separated by less than gap_ms milliseconds."""
        if not regions:
            return []

        merged: List[Tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_ms:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged
