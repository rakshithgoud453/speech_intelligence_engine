# Speech Intelligence Engine 🎙️⚡

An end-to-end, production-ready system that leverages **existing pretrained speech foundation models** (WavLM, HuBERT, Wav2Vec 2.0) to capture **"how something was said"** (prosody, rhythm, pauses, pitch dynamics, emotional inflection) rather than just "what words were said."

---

## The Core Concept

When speech is converted to text (`Speech -> ASR -> LLM`), 90% of paralinguistic nuance is lost:
- Pitch dynamics (rising vs falling inflection, question vs decisive assertion)
- Hesitation and pauses (hesitation count, silence duration ratio)
- Speaking rate and rhythm (rushed tempo vs deliberate cadence)
- Vocal energy dynamics (dynamic range, emphasis bursts)
- Layer-wise foundation model representations (early acoustic layers vs middle prosodic layers vs late semantic layers)

This engine **does not reinvent the wheel** by training a multi-billion parameter model from scratch. Instead, it extracts the rich internal transformer representations of existing self-supervised speech foundation models, combines them with signal-level prosodic profiling, and produces a structured **Speech Intelligence Layer** ready for LLMs, research experiments, and delivery style comparisons.

---

## Architecture Overview

```
                      Raw Audio Waveform (16kHz)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│ Pretrained Foundation Model  │        │   Signal-Level Prosodic Profiler │
│ (WavLM / HuBERT / Wav2Vec2)  │        │       (librosa, YIN, RMS)        │
└──────────────┬───────────────┘        └─────────────────┬────────────────┘
               │                                          │
       12 Transformer Layers                              │
    ┌──────────┼──────────┐                               │
    ▼          ▼          ▼                               ▼
Acoustic    Prosody    Semantic                 F0 Pitch Contour, Pauses,
(L0-L3)     (L4-L8)    (L9-L11)                 Speaking Rate, Dynamic Range
    └──────────┬──────────┘                               │
               ▼                                          ▼
     Energy-Weighted Pooling                     Prosody Feature Profile
               │                                          │
               └──────────────────┬───────────────────────┘
                                  ▼
                    Speech Intelligence Result
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
 768-d Vector Embeddings    Comparative Match     LLM Prompt Injection
   (Acoustic/Prosodic)     (Style / Concordance)    (JSON & Context)
```

---

## Quick Start

### 1. Run the Full Demo
Synthesizes 3 contrasting speech styles (hesitant, confident, questioning) and analyzes them end-to-end:

```bash
cd /Users/mac-0106/.gemini/antigravity/scratch/speech_intelligence_engine
/opt/homebrew/bin/python3.11 -m speech_intelligence_engine.cli demo
```

### 2. Analyze Any Audio File
Extract deep foundation representations and generate an interactive HTML dashboard:

```bash
/opt/homebrew/bin/python3.11 -m speech_intelligence_engine.cli analyze my_speech.wav \
    --model wavlm \
    --transcript "I don't know, maybe we should reconsider." \
    --html report.html \
    --json result.json \
    --save-embeddings embeddings.npz
```

### 3. Compare Delivery Between Two Audio Clips
Compare how two people spoke (tempo, pauses, pitch variation, and foundation layer similarity):

```bash
/opt/homebrew/bin/python3.11 -m speech_intelligence_engine.cli compare speaker1.wav speaker2.wav --model wavlm
```

---

## Python API Usage

```python
from speech_intelligence_engine import SpeechIntelligenceEngine

# 1. Initialize engine with preferred foundation model ('wavlm', 'hubert', or 'wav2vec2')
engine = SpeechIntelligenceEngine(model_type="wavlm")

# 2. Analyze speech file
result = engine.analyze(
    "demo_audio/hesitant_delivery.wav",
    transcript="Well... um... I really don't know... maybe tomorrow."
)

print(f"Delivery Style: {result.delivery_style_summary}")
print(f"Speaking Rate: {result.prosody_profile.estimated_speaking_rate_syllables_sec} syl/sec")
print(f"Pause Ratio: {result.prosody_profile.pause_ratio * 100:.1f}%")
print(f"Terminal Pitch Slope: {result.prosody_profile.terminal_pitch_slope}")

# 3. Access layer-wise & disentangled foundation embeddings
unified_emb = result.embeddings["unified_embedding"]    # (768,)
prosodic_emb = result.embeddings["prosodic_embedding"]  # (768,)
acoustic_emb = result.embeddings["acoustic_embedding"]  # (768,)

# 4. Inject paralinguistic context into an LLM
llm_prompt = result.llm_context_text
print(llm_prompt)

# 5. Compare two recordings for delivery concordance
comparison = engine.compare("speaker1.wav", "speaker2.wav")
print("Similarity verdict:", comparison["verdict"])
print("Prosodic layer similarity:", comparison["similarity_metrics"]["foundation_prosodic_layer_similarity"])
```

---

## Running Unit & Integration Tests

```bash
PYTHONPATH=/Users/mac-0106/.gemini/antigravity/scratch /opt/homebrew/bin/python3.11 -m unittest speech_intelligence_engine.tests.test_engine
```
