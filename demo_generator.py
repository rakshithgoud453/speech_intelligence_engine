"""
Demo Speech Generator.

Generates realistic audio samples illustrating contrasting delivery styles:
- Hesitant & Contemplative (slow, long pauses, uncertain)
- Confident & Decisive (rapid, continuous, falling cadence)
- Questioning & Expressive (rising inflection, high dynamic range)
"""

import os
import subprocess
from typing import Dict, List, Tuple


DEMO_PROFILES = [
    {
        "filename": "hesitant_delivery.wav",
        "text": "Well... um... I really don't know... maybe... we should think about it again... tomorrow.",
        "rate": 115,
        "description": "Hesitant & uncertain speech with extended pauses and slow tempo",
    },
    {
        "filename": "confident_delivery.wav",
        "text": "We are completely ready to proceed immediately with the system deployment today.",
        "rate": 210,
        "description": "Confident, brisk, decisive delivery with continuous cadence",
    },
    {
        "filename": "questioning_delivery.wav",
        "text": "Are you absolutely certain that this is the right direction for the architecture?",
        "rate": 160,
        "description": "Questioning intonation with prominent rising terminal pitch contour",
    },
]


def generate_demo_samples(output_dir: str) -> List[Dict[str, str]]:
    """
    Synthesize demo audio clips to output_dir using macOS native speech synthesis.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for item in DEMO_PROFILES:
        wav_path = os.path.join(output_dir, item["filename"])
        aiff_path = os.path.join(output_dir, item["filename"].replace(".wav", ".aiff"))

        # 1. Synthesize AIFF using macOS say
        cmd_say = [
            "say",
            "-r", str(item["rate"]),
            "-o", aiff_path,
            item["text"]
        ]
        subprocess.run(cmd_say, check=True)

        # 2. Convert to 16kHz 16-bit mono WAV using afconvert
        cmd_convert = [
            "afconvert",
            "-f", "WAVE",
            "-d", "LEI16@16000",
            "-c", "1",
            aiff_path,
            wav_path
        ]
        subprocess.run(cmd_convert, check=True)

        # Cleanup temporary AIFF
        if os.path.exists(aiff_path):
            os.remove(aiff_path)

        generated.append({
            "path": wav_path,
            "filename": item["filename"],
            "text": item["text"],
            "description": item["description"],
        })

    return generated


if __name__ == "__main__":
    out = os.path.expanduser("~/.gemini/antigravity/scratch/speech_intelligence_engine/demo_audio")
    res = generate_demo_samples(out)
    print(f"Generated {len(res)} demo audio files in {out}")
    for r in res:
        print(f" - {r['filename']}: {r['description']}")
