# SDK Reference — Speech Intelligence Engine

Complete API documentation for integrating with the Speech Learning Engine.

---

## Table of Contents

1. [SpeechLearningEngine (Orchestrator)](#1-speechlearningengine)
2. [AssimilationResult](#2-assimilationresult)
3. [EngineState](#3-enginestate)
4. [SpeechEncoder](#4-speechencoder)
5. [VoiceActivitySegmenter](#5-voiceactivitysegmenter)
6. [EpisodicMemory](#6-episodicmemory)
7. [PrototypeMemory](#7-prototypememory)
8. [RelationalMemory](#8-relationalmemory)
9. [ReplayBuffer](#9-replaybuffer)
10. [NoveltyDetector](#10-noveltydetector)
11. [GlobalStructureDiscovery](#11-globalstructurediscovery)
12. [Usage Patterns](#12-usage-patterns)

---

## 1. SpeechLearningEngine

The main entry point. Clients should only need this class for standard usage.

```python
from speech_intelligence_engine import SpeechLearningEngine
```

### Constructor

```python
engine = SpeechLearningEngine(
    model_type="hubert",        # "hubert" | "wavlm" | "wav2vec2"
    device=None,                # "cpu" | "cuda" | "mps" | None (auto-detect)
    replay_capacity=5000,       # Max entries in replay buffer
    max_prototypes=500,         # Max GNG prototype nodes
    discovery_min_episodes=100, # Min episodes before discovery can run
)
```

### Methods

#### `engine.assimilate(audio_input, sr=None, recording_id=None) → AssimilationResult`

Feed speech to the system. This is the primary learning method.

| Parameter | Type | Description |
|---|---|---|
| `audio_input` | `str` or `np.ndarray` | Path to `.wav` file, or a 1-D float32 numpy array. |
| `sr` | `int` or `None` | Sample rate. Required if `audio_input` is an array. Auto-detected from file. |
| `recording_id` | `str` or `None` | Optional external ID to track the source recording. |

```python
# From file
result = engine.assimilate("speech.wav")

# From numpy array
result = engine.assimilate(waveform_array, sr=16000, recording_id="rec_001")
```

#### `engine.run_discovery() → DiscoveryResult | None`

Run global structure discovery (UMAP + HDBSCAN). Heavyweight batch operation.
Returns `None` if not enough episodes have been collected yet.

```python
discovery = engine.run_discovery()
if discovery:
    print(f"Found {discovery.num_clusters} structures")
    print(f"Noise points: {discovery.noise_count}")
```

#### `engine.get_state() → EngineState`

Snapshot of the system's current knowledge state.

```python
state = engine.get_state()
print(state.total_episodes)       # How many speech segments experienced
print(state.total_prototypes)     # How many patterns discovered
print(state.total_transitions)    # How many temporal transitions recorded
print(state.overall_novelty_rate) # Fraction of inputs that were novel
print(state.is_novelty_burst)     # True if system is in a novelty burst
print(state.encoder_version)      # Current encoder version string
```

#### `engine.get_prototypes() → List[Prototype]`

Return all current prototype nodes from the GNG network.

```python
for proto in engine.get_prototypes():
    print(f"Pattern {proto.id[:8]}... — seen {proto.count} times")
    print(f"  First seen: {proto.first_seen}")
    print(f"  Last seen: {proto.last_seen}")
    print(f"  Vector shape: {proto.vector.shape}")
```

#### `engine.get_prototype_graph() → dict`

Return the relational memory as a serializable graph.

```python
graph = engine.get_prototype_graph()

# graph["nodes"]: list of {id, count, first_seen, last_seen, successors}
# graph["edges"]: list of {from, to, count, mean_delay_ms}

for node in graph["nodes"]:
    for succ in node["successors"]:
        print(f"{node['id'][:8]} → {succ['to'][:8]} "
              f"({succ['count']} times, {succ['mean_delay_ms']:.0f}ms delay)")
```

---

## 2. AssimilationResult

Returned by `engine.assimilate()`.

| Field | Type | Description |
|---|---|---|
| `recording_id` | `str \| None` | The recording ID you passed in. |
| `num_segments` | `int` | Number of voiced segments detected. |
| `num_episodes` | `int` | Number of episodes recorded (= num_segments). |
| `num_novel` | `int` | How many segments were flagged as novel. |
| `prototype_matches` | `List[str]` | Sequence of matched prototype IDs. |
| `novelty_rate` | `float` | Fraction of segments that were novel (0.0 – 1.0). |
| `duration_sec` | `float` | Total audio duration in seconds. |

---

## 3. EngineState

Returned by `engine.get_state()`.

| Field | Type | Description |
|---|---|---|
| `total_episodes` | `int` | Total speech segments experienced. |
| `total_prototypes` | `int` | Current number of GNG prototype nodes. |
| `total_transitions` | `int` | Total prototype-to-prototype transitions recorded. |
| `total_graph_edges` | `int` | Unique edges in the relational graph. |
| `replay_buffer_size` | `int` | Current entries in the replay buffer. |
| `replay_buffer_total_seen` | `int` | Total entries ever offered to the buffer. |
| `overall_novelty_rate` | `float` | Lifetime fraction of novel inputs. |
| `recent_novelty_rate` | `float` | Novelty rate over the last 50 inputs. |
| `is_novelty_burst` | `bool` | True if the system is in a sustained novelty burst. |
| `encoder_version` | `str` | Current encoder version string. |

---

## 4. SpeechEncoder

Low-level access to the foundation model. Use directly only if you need raw embeddings outside the learning loop.

```python
from speech_intelligence_engine import SpeechEncoder

encoder = SpeechEncoder(model_type="hubert")

# Encode a segment → temporal embedding (T, 768)
embedding = encoder.encode(waveform, sr=16000)
print(embedding.sequence.shape)    # (T, 768)
print(embedding.frame_times.shape) # (T,)
print(embedding.duration_sec)

# Pool to a single vector for retrieval
vec = embedding.pooled()           # (768,)

# Access all 12 transformer layers
layers = encoder.encode_all_layers(waveform, sr=16000)
# layers: List of 12 arrays, each (T, 768)
```

---

## 5. VoiceActivitySegmenter

Splits continuous audio into voiced speech segments.

```python
from speech_intelligence_engine import VoiceActivitySegmenter

segmenter = VoiceActivitySegmenter(
    sample_rate=16000,
    min_segment_ms=300,    # Discard segments shorter than 300ms
    min_silence_ms=200,    # A silence must be ≥200ms to split
)

segments = segmenter.segment(waveform, sr=16000)
for seg in segments:
    print(f"{seg.start_ms}ms – {seg.end_ms}ms ({seg.duration_ms}ms)")
    print(f"  Waveform shape: {seg.waveform.shape}")
```

---

## 6. EpisodicMemory

Append-only log. Direct access is rarely needed — the engine handles this.

```python
from speech_intelligence_engine import EpisodicMemory

memory = EpisodicMemory()

episode = memory.record(
    pooled_vector=vec,
    sequence_length=100,
    embedding_dim=768,
    model_version="hubert_base_v0",
)

print(memory.count())                    # Total episodes
print(memory.get_all_vectors().shape)    # (N, 768)
recent = memory.get_recent(10)           # Last 10 episodes
```

---

## 7. PrototypeMemory

Growing Neural Gas online prototype learner.

```python
from speech_intelligence_engine import PrototypeMemory

pm = PrototypeMemory(embedding_dim=768, max_nodes=500)

# Feed a vector — GNG update step
nearest_proto, is_novel = pm.update(vec)

# Query nearest prototypes
results = pm.nearest(query_vec, k=5)
for proto, distance in results:
    print(f"Pattern {proto.id[:8]} — distance {distance:.4f}")

# Get all prototypes
all_protos = pm.prototypes
print(f"Active prototypes: {pm.num_prototypes}")
```

---

## 8. RelationalMemory

Directed transition graph between prototypes.

```python
from speech_intelligence_engine import RelationalMemory

rm = RelationalMemory()

# Record a sequence of prototype activations
rm.record_sequence(
    prototype_ids=["A", "B", "C", "A"],
    frame_times_ms=[0, 500, 1200, 1800],
)

# Query transitions
successors = rm.get_successors("A")
# → [("B", count=1, mean_delay_ms=500.0)]

predecessors = rm.get_predecessors("A")
# → [("C", count=1, mean_delay_ms=600.0)]
```

---

## 9. ReplayBuffer

Reservoir sampling experience buffer.

```python
from speech_intelligence_engine import ReplayBuffer

buf = ReplayBuffer(capacity=5000, novelty_reserve_ratio=0.2)

buf.add(vector=vec, model_version="v0", is_novel=True)

batch = buf.sample(batch_size=32)
print(f"Buffer: {buf.size}/{buf.capacity} (seen {buf.total_seen})")
```

---

## 10. NoveltyDetector

Tracks novelty statistics and burst detection.

```python
from speech_intelligence_engine.discovery import NoveltyDetector

nd = NoveltyDetector(burst_window=50, burst_threshold=0.5)

event = nd.record(vector=vec, is_novel=True, distance=5.3)

print(nd.novelty_rate)         # Overall rate
print(nd.recent_novelty_rate)  # Recent window rate
print(nd.is_burst)             # True if sustained novel stream
```

---

## 11. GlobalStructureDiscovery

Periodic UMAP + HDBSCAN batch analysis.

```python
from speech_intelligence_engine.discovery import GlobalStructureDiscovery

gsd = GlobalStructureDiscovery(
    umap_n_components=10,
    hdbscan_min_cluster_size=20,
    min_vectors_for_discovery=100,
)

# vectors: (N, 768) matrix from episodic memory
result = gsd.discover(vectors)

if result:
    print(f"Clusters: {result.num_clusters}")
    print(f"Noise: {result.noise_count}")
    for label, centroid in result.centroids.items():
        print(f"  Cluster {label}: {result.cluster_sizes[label]} members")
```

---

## 12. Usage Patterns

### Pattern A: Batch Learning (Feed a Directory)

```python
import glob
from speech_intelligence_engine import SpeechLearningEngine

engine = SpeechLearningEngine(model_type="hubert")

# Feed all WAV files
for path in sorted(glob.glob("data/speeches/*.wav")):
    result = engine.assimilate(path)
    print(f"  {path}: {result.num_segments} segments, "
          f"{result.num_novel} novel")

# After batch, discover structures
discovery = engine.run_discovery()
state = engine.get_state()
print(f"\nLearned: {state.total_prototypes} patterns "
      f"from {state.total_episodes} episodes")
```

### Pattern B: Monitor Learning Progress

```python
for i, path in enumerate(audio_files):
    result = engine.assimilate(path)

    # Every 100 files, check state
    if (i + 1) % 100 == 0:
        state = engine.get_state()
        print(f"[{i+1}] Episodes={state.total_episodes}, "
              f"Prototypes={state.total_prototypes}, "
              f"NoveltyRate={state.overall_novelty_rate:.1%}")

        # Run discovery if we have enough data
        if state.total_episodes >= 100:
            engine.run_discovery()
```

### Pattern C: Export Knowledge Graph

```python
import json

graph = engine.get_prototype_graph()

# Save to JSON
with open("knowledge_graph.json", "w") as f:
    json.dump(graph, f, indent=2, default=str)

# Analyze: find the most common transition
edges = sorted(graph["edges"], key=lambda e: e["count"], reverse=True)
for edge in edges[:5]:
    print(f"{edge['from'][:8]} → {edge['to'][:8]}: "
          f"{edge['count']} times ({edge['mean_delay_ms']:.0f}ms)")
```

### Pattern D: Direct Access to Subsystems

If you need to bypass the orchestrator (e.g., for custom experiments):

```python
from speech_intelligence_engine import (
    SpeechEncoder,
    VoiceActivitySegmenter,
    PrototypeMemory,
    EpisodicMemory,
    RelationalMemory,
)

# Build your own pipeline
encoder = SpeechEncoder(model_type="wavlm")
segmenter = VoiceActivitySegmenter()
proto_mem = PrototypeMemory(embedding_dim=768)
episodic = EpisodicMemory()
relational = RelationalMemory()

# Custom loop
segments = segmenter.segment(waveform, sr=16000)
for seg in segments:
    emb = encoder.encode(seg.waveform, sr=seg.sample_rate)
    pooled = emb.pooled()
    episodic.record(pooled, emb.num_frames, emb.embedding_dim, encoder.version)
    proto, is_novel = proto_mem.update(pooled)
    # ... your custom logic here ...
```
