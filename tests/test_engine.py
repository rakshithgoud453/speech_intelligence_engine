"""
Unit and Integration Tests for Speech Intelligence Engine.
"""

import unittest
import numpy as np
import os
import tempfile
import soundfile as sf

from speech_intelligence_engine.prosody import ProsodicAnalyzer, ProsodyProfile
from speech_intelligence_engine.foundation import FoundationModelEmbedder
from speech_intelligence_engine.core import SpeechIntelligenceEngine
from speech_intelligence_engine.visualizer import SpeechVisualizer


class TestSpeechIntelligenceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate 2 seconds of 16kHz synthetic test signal:
        # 1 second tone (220 Hz) + 0.5s pause + 0.5s tone (330 Hz)
        cls.sr = 16000
        t1 = np.linspace(0, 1.0, cls.sr, endpoint=False)
        tone1 = 0.5 * np.sin(2 * np.pi * 220 * t1)
        silence = np.zeros(int(cls.sr * 0.5))
        t2 = np.linspace(0, 0.5, int(cls.sr * 0.5), endpoint=False)
        tone2 = 0.5 * np.sin(2 * np.pi * 330 * t2)

        cls.synthetic_audio = np.concatenate([tone1, silence, tone2]).astype(np.float32)

        # Temporary WAV file
        cls.tmp_dir = tempfile.mkdtemp()
        cls.test_wav_path = os.path.join(cls.tmp_dir, "test_synthetic.wav")
        sf.write(cls.test_wav_path, cls.synthetic_audio, cls.sr)

    def test_prosodic_analyzer(self):
        analyzer = ProsodicAnalyzer(sample_rate=self.sr)
        profile, time_series = analyzer.analyze(self.synthetic_audio, sr=self.sr)

        self.assertIsInstance(profile, ProsodyProfile)
        self.assertAlmostEqual(profile.duration_sec, 2.0, places=1)
        self.assertGreater(profile.pause_ratio, 0.15)  # should detect the 0.5s silence
        self.assertGreater(profile.mean_pitch_hz, 150.0)
        self.assertIn("times", time_series)
        self.assertIn("f0", time_series)
        self.assertIn("rms", time_series)

    def test_foundation_embedder(self):
        embedder = FoundationModelEmbedder(model_type="wavlm")
        layers, frame_times = embedder.extract_layers(self.synthetic_audio, sr=self.sr)

        # Should extract 12 transformer layers
        self.assertEqual(len(layers), 12)
        # Each layer should have 768 dimensions
        self.assertEqual(layers[0].shape[-1], 768)
        self.assertEqual(len(frame_times), layers[0].shape[1])

        # Test pooling & representation pack
        rep_pack = embedder.compute_representation_pack(layers)
        self.assertIn("unified_embedding", rep_pack)
        self.assertIn("prosodic_embedding", rep_pack)
        self.assertIn("acoustic_embedding", rep_pack)
        self.assertIn("semantic_embedding", rep_pack)
        self.assertEqual(rep_pack["unified_embedding"].shape, (768,))

    def test_end_to_end_engine(self):
        engine = SpeechIntelligenceEngine(model_type="wavlm")
        result = engine.analyze(self.test_wav_path, transcript="Testing synthetic audio.")

        self.assertIsNotNone(result.delivery_style_summary)
        self.assertGreater(len(result.dominant_cues), 0)
        self.assertIn("unified_embedding", result.embeddings)
        self.assertIn("Testing synthetic audio.", result.llm_context_text)

        # Test HTML report generation
        html_out = os.path.join(self.tmp_dir, "report.html")
        SpeechVisualizer.generate_html_report(result.to_dict(), output_path=html_out)
        self.assertTrue(os.path.exists(html_out))

    def test_speech_comparison(self):
        engine = SpeechIntelligenceEngine(model_type="wavlm")
        comp = engine.compare(self.test_wav_path, self.test_wav_path)
        # Comparing audio to itself should yield ~1.0 cosine similarity
        self.assertAlmostEqual(comp["similarity_metrics"]["foundation_unified_cosine_similarity"], 1.0, places=2)
        self.assertIn("verdict", comp)


if __name__ == "__main__":
    unittest.main()
