"""
Visualizer Module for Speech Intelligence Engine.

Generates self-contained, interactive HTML dashboards displaying:
- Time-domain waveform and loudness dynamics
- Pitch contour (F0) with identified hesitation/pause regions
- Foundation Model 12-layer activation heatmap
- Delivery style metrics and LLM prompt context
"""

import json
from typing import Dict, Any, Optional
import numpy as np


class SpeechVisualizer:
    """
    Renders interactive HTML reports using embedded SVG, Canvas, and CSS.
    """

    @staticmethod
    def generate_html_report(
        result_dict: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive, beautiful HTML dashboard report.
        """
        profile = result_dict.get("prosody_profile", {})
        style = result_dict.get("delivery_style_summary", "Analyzed Speech")
        cues = result_dict.get("dominant_cues", [])
        llm_text = result_dict.get("llm_context_text", "")
        audio_name = result_dict.get("audio_path") or "Audio Clip"

        cues_html = "".join([f"<li>{c}</li>" for c in cues])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Intelligence Dashboard - {audio_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-indigo: #818cf8;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-rose: #fb7185;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            padding: 30px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 28px;
            border-bottom: 1px solid var(--surface-border);
            padding-bottom: 18px;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 10px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 18px;
        }}
        .card-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text-main);
        }}
        .card-subtext {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .panel {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 24px;
        }}
        .panel-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--accent-blue);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        ul.cues-list {{
            list-style: none;
        }}
        ul.cues-list li {{
            position: relative;
            padding-left: 20px;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        ul.cues-list li::before {{
            content: "•";
            position: absolute;
            left: 4px;
            color: var(--accent-emerald);
            font-size: 18px;
        }}
        pre.llm-block {{
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            color: #e2e8f0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        .bar-container {{
            margin: 12px 0;
        }}
        .bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 4px;
        }}
        .bar-track {{
            background: #0f172a;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Speech Intelligence & Representation Profile</h1>
            <div class="subtitle">Acoustic & Foundation-Model Disentanglement | File: <code>{audio_name}</code></div>
            <div class="badge">Delivery: {style}</div>
        </header>

        <!-- Metric Summary Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-title">Duration & Speech Flow</div>
                <div class="card-value">{profile.get('duration_sec', 0.0)}s</div>
                <div class="card-subtext">{profile.get('speech_duration_sec', 0.0)}s voiced ({round(100 - profile.get('pause_ratio', 0)*100, 1)}%)</div>
            </div>
            <div class="card">
                <div class="card-title">Speaking Rate</div>
                <div class="card-value">{profile.get('estimated_speaking_rate_syllables_sec', 0.0)} <span style="font-size:14px; font-weight:normal;">syl/sec</span></div>
                <div class="card-subtext">Tempo: <strong>{profile.get('speech_tempo_label', 'moderate')}</strong></div>
            </div>
            <div class="card">
                <div class="card-title">Pause & Hesitation</div>
                <div class="card-value">{round(profile.get('pause_ratio', 0)*100, 1)}%</div>
                <div class="card-subtext">{profile.get('num_pauses', 0)} pauses (mean {profile.get('mean_pause_duration_sec', 0)}s)</div>
            </div>
            <div class="card">
                <div class="card-title">Pitch Range (F0)</div>
                <div class="card-value">{profile.get('pitch_range_semitones', 0.0)} <span style="font-size:14px; font-weight:normal;">semitones</span></div>
                <div class="card-subtext">Mean: {profile.get('mean_pitch_hz', 0)} Hz ({profile.get('pitch_variability_label', 'moderate')})</div>
            </div>
        </div>

        <!-- Paralinguistic Cues -->
        <div class="panel">
            <div class="panel-title">Auditory & Paralinguistic Cues (How it was said)</div>
            <ul class="cues-list">
                {cues_html}
            </ul>
        </div>

        <!-- Acoustic Breakdown -->
        <div class="panel">
            <div class="panel-title">Prosodic Dynamics Breakdown</div>
            
            <div class="bar-container">
                <div class="bar-label">
                    <span>Hesitation Index (Pause Ratio)</span>
                    <span>{round(profile.get('pause_ratio', 0)*100, 1)}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {min(100, profile.get('pause_ratio', 0)*150)}%; background: #fb7185;"></div>
                </div>
            </div>

            <div class="bar-container">
                <div class="bar-label">
                    <span>Dynamic Energy Variation (Emphasis)</span>
                    <span>{profile.get('dynamic_range_db', 0.0)} dB</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {min(100, (profile.get('dynamic_range_db', 0.0)/40.0)*100)}%; background: #38bdf8;"></div>
                </div>
            </div>

            <div class="bar-container">
                <div class="bar-label">
                    <span>Pitch Expressiveness</span>
                    <span>{profile.get('pitch_range_semitones', 0.0)} semitones</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {min(100, (profile.get('pitch_range_semitones', 0.0)/18.0)*100)}%; background: #34d399;"></div>
                </div>
            </div>
        </div>

        <!-- LLM Injection Prompt Section -->
        <div class="panel">
            <div class="panel-title">LLM Speech Intelligence Layer (Prompt Injection)</div>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 10px;">
                Feed this structured representation into an LLM along with the text transcription to preserve complete paralinguistic fidelity.
            </p>
            <pre class="llm-block">{llm_text}</pre>
        </div>
    </div>
</body>
</html>
"""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html
