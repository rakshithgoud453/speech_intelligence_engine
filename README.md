# Speech Intelligence Engine

A **continual learning system** that progressively develops an internal understanding of human speech. It does not classify, score, or judge speech. It *learns* from it.

---

## What This System Does

The engine consumes raw audio and builds an evolving internal representation of human speech through four cooperating subsystems:

1. **Encodes** speech into high-dimensional temporal sequences using pretrained foundation models (HuBERT, WavLM, Wav2Vec 2.0).
2. **Remembers** every experience in an immutable episodic log.
3. **Discovers** recurring patterns (prototypes) using the Growing Neural Gas algorithm and tracks how they transition into each other over time.
4. **Evolves** its own encoder through self-supervised learning with replay memory, becoming a better listener with each batch of audio it processes.

The **learning algorithm itself** is the product. Infrastructure exists only to feed it data and persist its state.

---

## What This System Does NOT Do

| It does NOT... | Why |
|---|---|
| Transcribe speech to text | Use Whisper or another ASR system for that. |
| Classify speech as "good" or "bad" | It has no concept of quality. It discovers structure. |
| Score speakers | There are no labels, ratings, or rankings. |
| Detect emotions by name | It discovers unnamed patterns, not "happy" or "sad." |
| Work as a real-time API | It is a batch/offline learning system, not a request/response server. |
| Replace an LLM | It produces embeddings and structural knowledge, not language. |

If you need inference, classification, or scoring — build a thin supervised layer **on top** of the learned representations later. The engine itself stays unsupervised.

---

## Architecture

```
                    HUMAN SPEECH
                         │
                         ▼
              ┌─────────────────────┐
              │  Voice Activity     │
              │  Segmenter (VAD)    │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  Speech Encoder θ   │
              │  (HuBERT/WavLM)     │
              └────────┬────────────┘
                       │
                       ▼
              Temporal Latent Sequence (T, 768)
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   ┌──────────┐  ┌───────────┐  ┌──────────┐
   │ Episodic │  │ Prototype │  │  Replay  │
   │ Memory   │  │ Memory    │  │  Buffer  │
   │ (log)    │  │ (GNG)     │  │(reservoir│
   └──────────┘  └─────┬─────┘  │sampling) │
                       │        └──────────┘
                       ▼
              ┌─────────────────────┐
              │ Relational Memory   │
              │ (transition graph)  │
              └─────────────────────┘

         Periodic:
         ┌─────────────────────────────────┐
         │  Global Discovery (UMAP+HDBSCAN)│
         │  Self-Supervised Training (EWC) │
         │  Stability Testing → Promote θ' │
         └─────────────────────────────────┘
```

---

## Installation

```bash
# Clone and install
git clone <repo-url>
cd speech_intelligence_engine
pip install -e .

# Optional: install structure discovery dependencies
pip install umap-learn hdbscan
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0
- 8 GB RAM minimum (16 GB recommended for large experiments)
- GPU optional but recommended for training phases

---

## Quick Start

```python
from speech_intelligence_engine import SpeechLearningEngine

# 1. Initialize the learning engine
engine = SpeechLearningEngine(model_type="hubert")

# 2. Feed it speech — it learns
engine.assimilate("recording_001.wav")
engine.assimilate("recording_002.wav")
engine.assimilate("recording_003.wav")
# ...keep feeding...

# 3. Check what the system knows
state = engine.get_state()
print(f"Episodes experienced: {state.total_episodes}")
print(f"Prototypes discovered: {state.total_prototypes}")
print(f"Transitions learned: {state.total_transitions}")
print(f"Novelty rate: {state.overall_novelty_rate:.1%}")

# 4. Periodically run structure discovery
discovery = engine.run_discovery()
if discovery:
    print(f"Clusters found: {discovery.num_clusters}")

# 5. Inspect the learned knowledge graph
graph = engine.get_prototype_graph()
for node in graph["nodes"][:5]:
    print(f"Pattern {node['id'][:8]}... — seen {node['count']} times")
```

---

## Project Structure

```
speech_intelligence_engine/
│
├── representation/          # Subsystem A: Sensory Input
│   ├── encoder.py           # HuBERT/WavLM/Wav2Vec2 → TemporalEmbedding(T, 768)
│   └── segmenter.py         # Energy-based VAD → List[Segment]
│
├── memory/                  # Subsystem B: Knowledge Storage
│   ├── episodic.py          # Immutable append-only experience log
│   ├── prototype.py         # Growing Neural Gas — online topology learning
│   ├── relational.py        # Directed transition graph between prototypes
│   └── replay.py            # Reservoir sampling experience buffer
│
├── discovery/               # Subsystem C: Pattern Discovery
│   ├── novelty.py           # Online novelty detection + burst alerting
│   └── global_structure.py  # UMAP + HDBSCAN batch clustering
│
├── learning/                # Subsystem D: Self-Improvement
│   ├── objectives.py        # Masked frame prediction (HuBERT-style SSL)
│   ├── trainer.py           # Candidate encoder training with EWC
│   └── stability.py         # Fixed-probe + Fisher-weighted validation
│
├── engine.py                # Orchestrator — the main entry point
├── prosody.py               # Signal-level prosodic analysis (F0, pauses, rhythm)
└── pyproject.toml           # Package configuration
```

---

## Algorithms & References

| Component | Algorithm | Reference |
|---|---|---|
| Prototype Memory | Growing Neural Gas (GNG) | Fritzke, B. (1995). *A Growing Neural Gas Network Learns Topologies.* NIPS 7. |
| Replay Buffer | Reservoir Sampling | Vitter, J.S. (1985). *Random Sampling with a Reservoir.* ACM TOMS. |
| Global Discovery | UMAP + HDBSCAN | McInnes & Healy (2017, 2018). |
| Training Objective | Masked Frame Prediction | Hsu et al. (2021). *HuBERT.* IEEE/ACM TASLP. |
| Stability Testing | EWC (Fisher Information) | Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting.* PNAS. |

---

cd /Users/mac-0106/Documents/my_projects/speech_intelligence_engine/python-ml-service

# Activate the virtual environment
source .venv/bin/activate

# (Optional) If you haven't installed the dependencies yet, run this first:
# pip install -r requirements.txt

# Start the gRPC server
python server.py

streamlit run playground.py

## License

Research use. See LICENSE file for details.
