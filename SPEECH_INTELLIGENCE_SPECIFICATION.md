# Speech Intelligence Engine — System Specification & Architectural Blueprint

> **Core Axiom**: This project is **NOT** a cascaded transcription/inference system (ASR -> LLM text pipeline). Tools, pipeline orchestrators, and web UIs are non-essential support tooling. The central mission of this system is to **natively learn, represent, and understand human speech** like a true Speech-LLM / Audio Foundation Engine.

---

## 1. Paradigm A vs. Paradigm B: Why Paradigm B is Real Speech Intelligence

### Paradigm A: Cascaded Tool Inference (What We Avoid as the End Goal)
$$\text{Speech Audio} \longrightarrow \text{ASR (Whisper)} \longrightarrow \text{Text String} \longrightarrow \text{Text LLM}$$

* **The Flaw**: Converting speech directly into flat text discards **>90% of paralinguistic and acoustic nuance**—including pauses, hesitation, rhythm, pitch inflections ($F_0$), vocal tension, cadence, and speaker dynamics. It treats speech as merely "typed words".
* **What it is**: An inference pipeline tool, not native speech intelligence.

### Paradigm B: Native End-to-End Speech-LLM & Representation Learner (Our Real Core System)
$$\text{Raw Audio Waveform (16kHz)} \longrightarrow \text{Audio Tokenization / Latent Projection} \longrightarrow \text{Multimodal Speech-LLM Backbone}$$

* **The Reality**: Human speech is processed directly as continuous acoustic vectors or neural audio codec tokens (e.g. Mimi, EnCodec, HuBERT/WavLM layers).
* **The Intelligence**: The model natively learns speech prosody, intonation, sarcasm, emotion, speaker identities, accent, and conversational turns directly within its neural weights.

---

## 2. System Architecture: How the Learner Consumes Speech

```
                             Raw Audio Waveform (16 kHz)
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
┌───────────────────────────┐                           ┌───────────────────────────┐
│  Discrete Audio Codec /   │                           │  Self-Supervised Audio    │
│  Latent Projection Layer  │                           │  Encoder (WavLM/HuBERT)   │
└────────────┬──────────────┘                           └────────────┬──────────────┘
             │                                                       │
             ▼                                                       ▼
┌───────────────────────────┐                           ┌───────────────────────────┐
│ Audio Tokenizer           │                           │ Intermediate Layer        │
│ (Continuous/Discrete)     │                           │ Representation Extractor  │
└────────────┬──────────────┘                           └────────────┬──────────────┘
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         ▼
                       ┌───────────────────────────────────┐
                       │  Multimodal Transformer Backbone  │
                       │  (Native Speech-LLM Core)         │
                       └─────────────────┬─────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│ Continual Learning & Memory Loop  │       │ Direct Native Speech Reasoning    │
│ (Episodic, Prototype, Replay)     │       │ & Audio Synthesis / Response      │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

---

## 3. Market Landscape of Native Speech Foundation Models

| Model / Architecture | Organization | Technical Mechanism |
| :--- | :--- | :--- |
| **Qwen2-Audio / Qwen2.5-Audio** | Alibaba | Multimodal audio encoder (Whisper-large-v3) projected directly into LLM token space for speech QA and understanding without cascading. |
| **Ultravox** | Fixie.ai | Multimodal projector ("UltravoxProjector") mapping continuous audio latent vectors directly into Llama 3 embedding space. |
| **Moshi** | Kyutai | Full-duplex speech-to-speech model using the **Mimi** neural audio codec and dual-stream inner monologue text token generation. |
| **Spirit LM** | Meta | Interleaved text and speech tokens using phonetic units (HuBERT) and explicit pitch/style units for expressive prosody. |
| **AudioPaLM** | Google | Unified text and speech transformer with shared audio-text vocabulary. |
| **GPT-4o Realtime / Gemini 2.0** | OpenAI / Google | Proprietary end-to-end native audio foundation models with prosody, intonation, and emotion awareness. |

---

## 4. Current Audit of Our Repository Codebase

In our python runtime (`python-ml-service` and `playground.py`), we currently run:

| Model | Library | Role | Status |
| :--- | :--- | :--- | :--- |
| **Faster-Whisper (`small`)** | `faster-whisper` | Raw ASR Transcription | Inference tool |
| **PyAnnote Audio (`2.1`)** | `pyannote.audio` | Speaker Diarization | Inference tool |

> **Note**: Our repository also contains initial research components in `foundation.py`, `prosody.py`, and `engine.py` for extracting intermediate WavLM/HuBERT layer representations, frame-level pitch ($F_0$), and dynamic energy pooling.

---

## 5. Blueprint: Moving to a True Speech-LLM System

1. **Native Speech Tokenization**: Connect raw audio to a continuous projector (like Ultravox or Qwen2-Audio style encoder) or discrete audio codec (Mimi / EnCodec).
2. **Intermediate SSL Feature Extraction**: Use `foundation.py` to extract Layers 4–8 (prosody/phonetic) and 9–11 (semantic context) from SSL models (WavLM / HuBERT).
3. **Continual Memory & Representation Evolution**:
   - **Episodic Memory**: Store latent speech experiences across time.
   - **Prototype Memory**: Unsupervised discovery of recurring speech structures using density discovery (HDBSCAN).
   - **Replay Memory**: Maintain replay buffers during continual learning to prevent catastrophic forgetting.
4. **Direct Audio Reasoning**: Query the Speech Engine directly about acoustic context, vocal shift, hesitation, speaker intent, and emotional nuance.
