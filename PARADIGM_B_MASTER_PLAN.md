# Paradigm B Master Implementation Plan: Native Speech Intelligence Engine

> **Mission**: Build an autonomous, self-supervised AI system that continuously consumes raw speech, builds evolving internal acoustic/prosodic/semantic representations, and connects directly to a multimodal Speech-LLM backbone—offloading heavy compute to Google Colab GPUs synced with 5 TB Google Drive storage (`rakshithgoud453@gmail.com`).

---

## 1. System Architecture Overview

```
                                  Raw Audio Waveform (16 kHz)
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         ┌──────────────────────────┐                    ┌──────────────────────────┐
         │ Voice Activity Segmenter │                    │  Signal Prosody Profiler │
         │ (VAD / Energy Threshold) │                    │ (F0 Pitch, RMS, Tempo)   │
         └─────────────┬────────────┘                    └─────────────┬────────────┘
                       │                                               │
                       ▼                                               ▼
         ┌──────────────────────────┐                    ┌──────────────────────────┐
         │  SSL Speech Encoder      │                    │ Frame-Level Acoustic     │
         │  (WavLM / HuBERT)        │                    │ Feature Vector           │
         └─────────────┬────────────┘                    └─────────────┬────────────┘
                       │                                               │
        ┌──────────────┼──────────────┐                                │
        ▼              ▼              ▼                                │
   Timbre/Acoustic  Prosodic       Semantic                            │
     (L0 - L3)      (L4 - L8)     (L9 - L11)                           │
        └──────────────┬──────────────┘                                │
                       ▼                                               ▼
             Energy-Weighted Temporal Pooling                          │
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               ▼
                             ┌───────────────────────────────────┐
                             │  Speech Learning Engine (engine.py)│
                             └─────────────────┬─────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
┌──────────────────┐                 ┌──────────────────┐                  ┌──────────────────┐
│ Episodic Memory  │                 │ Prototype Memory │                  │ Relational Graph │
│ (Immutable Traj) │                 │ (Dynamic GNG)    │                  │ (Transitions A→B)│
└────────┬─────────┘                 └────────┬─────────┘                  └────────┬─────────┘
         │                                    │                                     │
         └────────────────────────────────────┼─────────────────────────────────────┘
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Replay Buffer & HDBSCAN  │
                                 │ Structure Discovery      │
                                 └────────────┬─────────────┘
                                              │
                                              ▼
                             ┌───────────────────────────────────┐
                             │ Multimodal Speech-LLM Projector   │
                             │ (Ultravox/Qwen2 Adapter via LoRA) │
                             └───────────────────────────────────┘
```

---

## 2. Core Architecture Subsystems (In-Repo Inventory)

| Subsystem | Module Path | Purpose |
| :--- | :--- | :--- |
| **Representation Engine** | `representation/encoder.py` | Extracts 12 hidden states from `microsoft/wavlm-base-plus` or `facebook/hubert-base-ls960`. Pools vectors temporally across frames. |
| **Audio Segmenter** | `representation/segmenter.py` | Performs Voice Activity Detection (VAD) to split continuous speech into meaningful acoustic segments. |
| **Episodic Memory** | `memory/episodic.py` | Immutable database of historical speech experiences and latent trajectories. |
| **Prototype Memory** | `memory/prototype.py` | Growing Neural Gas (GNG) topological network that discovers recurring speech structures without human labels. |
| **Relational Memory** | `memory/relational.py` | Directed transition graph ($A \rightarrow B \rightarrow C$) capturing temporal patterns, hesitation pauses, and delivery sequences. |
| **Replay & Discovery** | `memory/replay.py` & `discovery/` | Reservoir sampling + UMAP/HDBSCAN global structure discovery to prevent catastrophic forgetting. |
| **Engine Orchestrator** | `engine.py` | Central `SpeechLearningEngine` state machine coordinating the complete continual assimilation loop. |
| **Colab GPU Trainer** | `notebooks/speech_intelligence_colab_trainer.ipynb` | Offloaded heavy training & SSL feature extraction running on Colab GPUs with 5 TB Google Drive sync. |

---

## 3. Detailed Phase-by-Phase Execution Plan

### Phase 1: Local Engine Verification & Assimilation (Local Mac)
* **Goal**: Verify `SpeechLearningEngine` assimilation loop end-to-end on local audio clips.
* **Tasks**:
  1. Initialize `SpeechLearningEngine(model_type="wavlm")`.
  2. Ingest test speech recordings using `engine.assimilate(audio_path)`.
  3. Validate prototype allocation, novelty score generation, and relational graph edge creation.
  4. Ensure `EngineState` metrics correctly track episodes, prototypes, and transition density.

### Phase 2: Remote GPU Data Ingestion & Structure Discovery (Google Colab + 5 TB Drive)
* **Goal**: Offload batch processing of 100+ speech recordings to Colab GPUs.
* **Tasks**:
  1. Open `notebooks/speech_intelligence_colab_trainer.ipynb` on Google Colab signed into `rakshithgoud453@gmail.com`.
  2. Batch extract SSL layer representations (Layers 4–8 prosodic, Layers 9–11 semantic) using Colab GPU.
  3. Run UMAP dimension reduction + HDBSCAN density clustering across all stored speech vectors.
  4. Save discovered prototype centroids and segment manifests directly to Google Drive (`/content/drive/MyDrive/speech_intelligence_engine/checkpoints/`).

### Phase 3: Speech-LLM Multimodal Projector Training
* **Goal**: Connect speech latent representations directly to an LLM token space so the LLM can "listen" to raw audio embeddings natively.
* **Tasks**:
  1. Train a lightweight Multimodal Adapter (Linear Projector / 2-layer MLP) mapping 768-d WavLM/HuBERT speech vectors into the token embedding space of an LLM (e.g. Llama 3 or Qwen2).
  2. Use LoRA (Low-Rank Adaptation via `peft`) on Colab GPU to keep training lightweight and fast.
  3. Save adapter weights to 5 TB Google Drive.

### Phase 4: Interactive Model Explorer & Evaluation UI
* **Goal**: Create an interactive environment to test native speech understanding.
* **Tasks**:
  1. Update `python-ml-service/playground.py` to load the trained native speech engine.
  2. Upload audio files and directly query the engine about acoustic delivery, hesitation, speaker transitions, and sentiment shifts.
