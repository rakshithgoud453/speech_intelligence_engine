"""
Command Line Interface for Speech Intelligence Engine.
"""

import argparse
import sys
import os
import json
from pprint import pprint

from .core import SpeechIntelligenceEngine
from .visualizer import SpeechVisualizer
from .demo_generator import generate_demo_samples


def main():
    parser = argparse.ArgumentParser(
        description="Speech Intelligence Engine: Extract 'how it was said' from speech foundation models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze an audio file for speech intelligence and prosody.")
    p_analyze.add_argument("audio", type=str, help="Path to input audio file (.wav, .mp3, etc.)")
    p_analyze.add_argument("--model", type=str, default="wavlm", choices=["wavlm", "hubert", "wav2vec2"],
                           help="Pretrained foundation model to use (default: wavlm)")
    p_analyze.add_argument("--transcript", type=str, default=None, help="Spoken text transcript (optional)")
    p_analyze.add_argument("--html", type=str, default=None, help="Save interactive HTML dashboard to path")
    p_analyze.add_argument("--json", type=str, default=None, help="Save structured JSON result to path")
    p_analyze.add_argument("--save-embeddings", type=str, default=None,
                           help="Save extracted foundation embeddings to .npz file")

    # Subcommand: compare
    p_compare = subparsers.add_parser("compare", help="Compare delivery style and foundation representations between two audio clips.")
    p_compare.add_argument("audio1", type=str, help="First audio file")
    p_compare.add_argument("audio2", type=str, help="Second audio file")
    p_compare.add_argument("--model", type=str, default="wavlm", choices=["wavlm", "hubert", "wav2vec2"],
                           help="Pretrained foundation model to use (default: wavlm)")

    # Subcommand: demo
    p_demo = subparsers.add_parser("demo", help="Generate synthetic speech samples and demonstrate end-to-end extraction & comparison.")
    p_demo.add_argument("--model", type=str, default="wavlm", choices=["wavlm", "hubert", "wav2vec2"],
                        help="Foundation model to use for demo (default: wavlm)")

    args = parser.parse_args()

    if args.command == "analyze":
        print(f"[*] Initializing Speech Intelligence Engine with model '{args.model}'...")
        engine = SpeechIntelligenceEngine(model_type=args.model)
        print(f"[*] Analyzing audio: {args.audio}...")
        result = engine.analyze(args.audio, transcript=args.transcript)

        print("\n" + "="*60)
        print(f"DELIVERY STYLE: {result.delivery_style_summary}")
        print("="*60)
        print(f"Duration: {result.duration_sec:.2f}s (Voiced: {result.prosody_profile.speech_duration_sec:.2f}s)")
        print(f"Speaking Rate: {result.prosody_profile.estimated_speaking_rate_syllables_sec} syl/sec ({result.prosody_profile.speech_tempo_label})")
        print(f"Pauses: {result.prosody_profile.num_pauses} pauses, {result.prosody_profile.pause_ratio*100:.1f}% pause ratio")
        print(f"Pitch Dynamics: {result.prosody_profile.pitch_variability_label} ({result.prosody_profile.pitch_range_semitones:.1f} semitones)")
        print(f"Terminal Pitch Slope: {result.prosody_profile.terminal_pitch_slope:.2f} ({'Rising' if result.prosody_profile.terminal_pitch_slope > 0.5 else 'Falling' if result.prosody_profile.terminal_pitch_slope < -0.5 else 'Flat'})")
        print(f"Foundation Vector: 768-dim (Unified WavLM embedding)")

        print("\n[*] LLM Context Block:")
        print("-" * 50)
        print(result.llm_context_text)
        print("-" * 50)

        if args.html:
            SpeechVisualizer.generate_html_report(result.to_dict(), output_path=args.html)
            print(f"[+] Saved interactive HTML dashboard to: {args.html}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"[+] Saved structured JSON result to: {args.json}")

        if args.save_embeddings:
            import numpy as np
            np.savez_compressed(args.save_embeddings, **result.embeddings)
            print(f"[+] Saved foundation embeddings to: {args.save_embeddings}")

    elif args.command == "compare":
        print(f"[*] Initializing Speech Intelligence Engine with model '{args.model}'...")
        engine = SpeechIntelligenceEngine(model_type=args.model)
        print(f"[*] Comparing delivery styles:\n  - Clip 1: {args.audio1}\n  - Clip 2: {args.audio2}")
        comp = engine.compare(args.audio1, args.audio2)

        print("\n" + "="*60)
        print(f"VERDICT: {comp['verdict']}")
        print("="*60)
        print("\nClip 1 Profile:")
        for k, v in comp["clip_1"].items():
            print(f"  {k}: {v}")
        print("\nClip 2 Profile:")
        for k, v in comp["clip_2"].items():
            print(f"  {k}: {v}")
        print("\nSimilarity Metrics:")
        for k, v in comp["similarity_metrics"].items():
            print(f"  {k}: {v}")

    elif args.command == "demo":
        print("="*70)
        print("RUNNING COMPLETE SPEECH INTELLIGENCE DEMONSTRATION")
        print("="*70)
        demo_dir = os.path.expanduser("~/.gemini/antigravity/scratch/speech_intelligence_engine/demo_audio")
        print(f"[*] Step 1: Synthesizing demo audio clips to {demo_dir}...")
        samples = generate_demo_samples(demo_dir)
        for s in samples:
            print(f"  [+] Created {s['filename']}: \"{s['text']}\"")

        print(f"\n[*] Step 2: Loading Speech Intelligence Engine ('{args.model}')...")
        engine = SpeechIntelligenceEngine(model_type=args.model)

        analyzed_results = []
        for s in samples:
            print(f"\n---> Analyzing {s['filename']} ({s['description']})...")
            res = engine.analyze(s["path"], transcript=s["text"])
            analyzed_results.append(res)
            print(f"     Identified Delivery: {res.delivery_style_summary}")
            print(f"     Speaking Rate: {res.prosody_profile.estimated_speaking_rate_syllables_sec} syl/sec")
            print(f"     Pause Ratio: {res.prosody_profile.pause_ratio*100:.1f}%")
            print(f"     Terminal Pitch Contour: {res.prosody_profile.terminal_pitch_slope:.2f}")

            # Generate individual HTML dashboard
            html_out = s["path"].replace(".wav", "_dashboard.html")
            SpeechVisualizer.generate_html_report(res.to_dict(), output_path=html_out)
            print(f"     [Report] Dashboard saved to {html_out}")

        print("\n" + "="*70)
        print("[*] Step 3: Comparing Delivery Styles Between Contrasting Clips...")
        print("="*70)
        comp_hesitant_vs_confident = engine.compare(samples[0]["path"], samples[1]["path"])
        print("\nComparison 1: Hesitant Speaker vs Confident Speaker")
        print(f"Verdict: {comp_hesitant_vs_confident['verdict']}")
        print(f"Foundation Prosodic Layer Similarity: {comp_hesitant_vs_confident['similarity_metrics']['foundation_prosodic_layer_similarity']}")
        print(f"Speaking Rate Difference: {comp_hesitant_vs_confident['similarity_metrics']['tempo_difference_syl_sec']} syl/sec")
        print(f"Pause Ratio Difference: {comp_hesitant_vs_confident['similarity_metrics']['pause_ratio_difference_percent']}%")

        comp_confident_vs_questioning = engine.compare(samples[1]["path"], samples[2]["path"])
        print("\nComparison 2: Confident Speaker vs Questioning Speaker")
        print(f"Verdict: {comp_confident_vs_questioning['verdict']}")
        print(f"Foundation Prosodic Layer Similarity: {comp_confident_vs_questioning['similarity_metrics']['foundation_prosodic_layer_similarity']}")
        print(f"Pitch Range Difference: {comp_confident_vs_questioning['similarity_metrics']['pitch_range_difference_semitones']} st")

        print("\n" + "="*70)
        print("[+] DEMO COMPLETE: The Speech Intelligence Engine successfully extracted")
        print("    the foundation representations and paralinguistic profiles.")
        print("="*70)


if __name__ == "__main__":
    main()
