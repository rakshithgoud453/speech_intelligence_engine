"""
Quickstart / Example script for Speech Intelligence Engine.

Usage:
    # 1. Run automated demo with simulated speech patterns:
    python example.py

    # 2. Or pass real audio files:
    python example.py path/to/sample1.wav path/to/sample2.wav
"""

import sys
import os
import numpy as np

# Ensure parent directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from speech_intelligence_engine import SpeechLearningEngine


def generate_synthetic_speech_burst(freq: float, duration_sec: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Generate harmonic audio bursts to simulate speech vocalization for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental frequency + harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
        + 0.15 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    # Amplitude envelope
    envelope = np.hanning(len(t))
    return (audio * envelope).astype(np.float32)


def main():
    print("=" * 60)
    print("🧠 Speech Intelligence Engine — Continual Learning Demo")
    print("=" * 60)

    # 1. Initialize Engine
    print("\n[1/4] Initializing SpeechLearningEngine (HuBERT representation)...")
    engine = SpeechLearningEngine(
        model_type="hubert",
        replay_capacity=500,
        max_prototypes=100,
        discovery_min_episodes=5,
    )
    print("✓ Engine initialized successfully.")

    # 2. Ingest Audio
    audio_files = sys.argv[1:] if len(sys.argv) > 1 else []

    if audio_files:
        print(f"\n[2/4] Assimilating {len(audio_files)} provided audio files...")
        for i, file_path in enumerate(audio_files, 1):
            if not os.path.exists(file_path):
                print(f"  [!] Warning: File '{file_path}' not found, skipping.")
                continue
            print(f"  → Ingesting [{i}/{len(audio_files)}]: {file_path}")
            res = engine.assimilate(file_path, recording_id=os.path.basename(file_path))
            print(
                f"    ✓ Duration: {res.duration_sec:.2f}s | Segments: {res.num_segments} | "
                f"Episodes: {res.num_episodes} | Novel: {res.num_novel} (Novelty: {res.novelty_rate:.1%})"
            )
    else:
        print("\n[2/4] No audio files provided. Simulating speech episodes with varying acoustic patterns...")
        patterns = [
            ("Pattern-A (150 Hz Base)", 150.0),
            ("Pattern-A (150 Hz Base)", 150.0),
            ("Pattern-B (250 Hz Higher)", 250.0),
            ("Pattern-A (150 Hz Base)", 150.0),
            ("Pattern-C (350 Hz High)", 350.0),
            ("Pattern-B (250 Hz Higher)", 250.0),
        ]
        sr = 16000
        for i, (name, freq) in enumerate(patterns, 1):
            audio = generate_synthetic_speech_burst(freq=freq, duration_sec=1.5, sr=sr)
            print(f"  → Ingesting [{i}/{len(patterns)}] {name}...")
            res = engine.assimilate(audio, sr=sr, recording_id=f"sim_{i}")
            print(
                f"    ✓ Duration: {res.duration_sec:.2f}s | Segments: {res.num_segments} | "
                f"Episodes: {res.num_episodes} | Novel: {res.num_novel} (Novelty: {res.novelty_rate:.1%})"
            )

    # 3. Check System State
    print("\n[3/4] Current Engine State & Memory:")
    state = engine.get_state()
    print(f"  • Total Episodes Experienced: {state.total_episodes}")
    print(f"  • Total Prototypes Discovered: {state.total_prototypes}")
    print(f"  • Learned Graph Transitions:   {state.total_transitions}")
    print(f"  • Replay Buffer Size:          {state.replay_buffer_size} / {state.replay_buffer_total_seen} seen")
    print(f"  • Overall Novelty Rate:        {state.overall_novelty_rate:.1%}")
    print(f"  • Novelty Burst Detected:      {state.is_novelty_burst}")

    # 4. Inspect Learned Prototype Graph
    print("\n[4/4] Learned Prototype Graph Snapshot:")
    graph = engine.get_prototype_graph()
    print(f"  • Node Count: {len(graph['nodes'])} | Edge Count: {len(graph['edges'])}")
    for node in graph["nodes"][:5]:
        print(f"    - Prototype {node['id'][:8]}... | Frequency count: {node['count']}")

    print("\n" + "=" * 60)
    print("✨ Demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
