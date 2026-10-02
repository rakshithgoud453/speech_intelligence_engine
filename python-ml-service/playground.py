import sys
import os
import json
import time
import numpy as np
import streamlit as st

# Add repository root and parent directory to python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(repo_root, ".."))

for p in [repo_root, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from speech_intelligence_engine import SpeechLearningEngine

st.set_page_config(
    page_title="Speech Intelligence Engine — Live Learner Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Speech Intelligence Engine — Live Learner Dashboard")
st.markdown(
    "**Paradigm B: Native Continual Speech Learning System.** "
    "Inspect how raw 16kHz audio is processed through 12-layer HuBERT/WavLM hidden representations, "
    "disentangled into prosodic/phonetic/semantic layers, and organized into an evolving memory graph."
)

@st.cache_resource
def get_native_engine():
    return SpeechLearningEngine(model_type="hubert", replay_capacity=1000, max_prototypes=100)

engine = get_native_engine()

# Sidebar: Live Telemetry & Control Panel
st.sidebar.header("⚙️ Learner Telemetry & Options")
st.sidebar.markdown(f"**Encoder Core**: `{engine.encoder.model_type.upper()}` (12 Layers)")
st.sidebar.markdown(f"**Vector Dimension**: `768-d` (Energy-Weighted Pooling)")

if st.sidebar.button("🔄 Reset Engine State"):
    st.cache_resource.clear()
    st.rerun()

# ─── SECTION 1: LIVE LEARNER METRICS ──────────────────────────────────────────────
st.subheader("1. Live Engine Memory & Learning State")

state = engine.get_state()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Episodes Learned", state.total_episodes, help="Total speech instances experienced")
col2.metric("Discovered Prototypes", state.total_prototypes, help="Unsupervised acoustic sound clusters discovered")
col3.metric("Relational Transitions", state.total_transitions, help="Learned temporal sequence transitions (Pattern A → B)")
col4.metric("Replay Buffer Size", f"{state.replay_buffer_size} / {state.replay_buffer_total_seen}", help="Buffer capacity for continual learning")
col5.metric("Novelty Rate", f"{state.overall_novelty_rate:.1%}", help="Ratio of unseen acoustic patterns encountered")

st.divider()

# ─── SECTION 2: LIVE AUDIO ASSIMILATION & SENSORY INSPECTOR ────────────────────────
st.subheader("2. Live Speech Assimilation & SSL Layer Disentanglement")
st.markdown(
    "Experience raw speech sequentially. When speech enters the engine, it is segmented by Voice Activity Detection (VAD), "
    "passed through the SSL sensory cortex, and evaluated against prototype memory."
)

tab1, tab2 = st.parse_config if hasattr(st, "parse_config") else (st.tabs(["🎙️ Ingest & Assimilate Speech", "☁️ Remote Colab Training Status"]) if hasattr(st, "tabs") else (None, None))

if tab1:
    with tab1:
        input_type = st.radio("Select Input Source:", ["Generate Synthetic Speech Pattern (Demo)", "Upload Audio File (WAV/MP3)"], horizontal=True)

        audio_data = None
        sr = 16000
        recording_id = "sample"

        if "Upload" in input_type:
            uploaded_file = st.file_uploader("Choose a WAV or MP3 audio file", type=["wav", "mp3"])
            if uploaded_file is not None:
                st.audio(uploaded_file, format="audio/wav")
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                audio_data = temp_path
                recording_id = uploaded_file.name
        else:
            pattern_freq = st.slider("Simulated Pitch Frequency (Hz):", min_value=100.0, max_value=400.0, value=180.0, step=10.0)
            if st.button("Generate & Play Simulated Burst"):
                t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
                sig = (0.5 * np.sin(2 * np.pi * pattern_freq * t) + 0.2 * np.sin(2 * np.pi * (pattern_freq * 2) * t)).astype(np.float32)
                temp_path = f"/tmp/sim_{int(pattern_freq)}hz.wav"
                import soundfile as sf
                sf.write(temp_path, sig, sr)
                audio_data = temp_path
                recording_id = f"sim_{int(pattern_freq)}hz"
                st.audio(temp_path, format="audio/wav")

        if audio_data is not None:
            if st.button("⚡ Assimilate Speech into Engine Memory"):
                with st.spinner("Processing speech through HuBERT sensory cortex & updating memory..."):
                    res = engine.assimilate(audio_data, recording_id=recording_id)

                    st.success(f"✓ Assimilation Complete! Ingested '{recording_id}' ({res.duration_sec:.2f} seconds)")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Voiced Segments", res.num_segments)
                    r2.metric("New Episodes Stored", res.num_episodes)
                    r3.metric("Novel Patterns Births", res.num_novel)
                    r4.metric("Novelty Ratio", f"{res.novelty_rate:.1%}")

                    st.markdown("#### 🔍 Layer Disentanglement Mapping (SSL Sensory Cortex)")
                    st.markdown("""
                    - **Layers 0–3 (Timbre & Spectral)**: Vocal tract resonance, timbre brightness, low-level spectral structure.
                    - **Layers 4–8 (Prosody & Phonetics)**: Pitch inflection ($F_0$), syllabic rhythm, stress points, pauses.
                    - **Layers 9–11 (Semantic Context)**: High-level linguistic patterns and utterance context.
                    """)

                    st.markdown("#### 🎯 Prototype Matching Results per Segment")
                    if res.prototype_matches:
                        st.json({"matched_prototypes": res.prototype_matches})
                    else:
                        st.info("No existing prototype matched — created novel acoustic prototypes!")

with tab2:
    st.markdown("#### ☁️ Remote Colab & Drive Checkpoint Inspector")
    st.markdown("Inspect training execution, model checkpoint weights (`.pt`), and telemetry stored on Google Drive or local workspace.")

    checkpoint_dir = os.path.abspath(os.path.join(repo_root, "checkpoints"))
    local_checkpoints = []
    if os.path.exists(checkpoint_dir):
        local_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

    c_col1, c_col2 = st.columns([1, 1])
    with c_col1:
        st.markdown("**Load Saved Model Checkpoint (.pt):**")
        uploaded_ckpt = st.file_uploader("Upload PyTorch Checkpoint (.pt)", type=["pt"])

        selected_ckpt_path = None
        if uploaded_ckpt is not None:
            temp_ckpt = f"/tmp/{uploaded_ckpt.name}"
            with open(temp_ckpt, "wb") as f:
                f.write(uploaded_ckpt.getbuffer())
            selected_ckpt_path = temp_ckpt
        elif local_checkpoints:
            selected_ckpt_path = st.selectbox("Select Checkpoint from Workspace:", local_checkpoints)

        if selected_ckpt_path and os.path.exists(selected_ckpt_path):
            try:
                import torch
                ckpt = torch.load(selected_ckpt_path, map_location="cpu")
                st.success(f"✓ Loaded Checkpoint: `{os.path.basename(selected_ckpt_path)}`")

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Training Step", ckpt.get("step", "N/A"))
                m_col2.metric("Saved Keys", len(ckpt.get("model_state_dict", {})))

                metrics = ckpt.get("metrics", {})
                if metrics:
                    m_col3.metric("Episodes", metrics.get("episodes", "N/A"))
                    st.json(metrics)

                st.markdown("**State Dict Tensors:**")
                st.json({k: list(v.shape) for k, v in ckpt.get("model_state_dict", {}).items()})
            except Exception as ex:
                st.error(f"Failed to inspect checkpoint: {ex}")
        else:
            st.info("No checkpoint file selected. Run Colab training or upload a `.pt` file above.")

    with c_col2:
        st.markdown("**Google Drive Remote Telemetry Status (`status.json`):**")
        drive_status_path = "/content/drive/MyDrive/speech_intelligence_engine/outputs/status.json"
        local_status_path = os.path.abspath(os.path.join(repo_root, "outputs", "status.json"))

        active_status_path = local_status_path if os.path.exists(local_status_path) else drive_status_path
        if os.path.exists(active_status_path):
            with open(active_status_path, "r") as f:
                status_data = json.load(f)
            st.success(f"✓ Training Telemetry Active (`{active_status_path}`)!")
            st.json(status_data)
        else:
            st.info("Colab telemetry status file will appear here automatically when Step 6 finishes.")

st.divider()

# ─── SECTION 3: PROTOTYPE TOPOLOGICAL GRAPH ─────────────────────────────────────────
st.subheader("3. Discovered Acoustic Prototypes & Transition Graph")
st.markdown("Visualizes the Growing Neural Gas (GNG) topological network of learned speech sound structures.")

graph = engine.get_prototype_graph()

if graph["nodes"]:
    g_col1, g_col2 = st.columns([1, 1])
    with g_col1:
        st.markdown(f"**Discovered Prototype Nodes ({len(graph['nodes'])} total):**")
        nodes_display = [
            {
                "Prototype ID": n["id"][:12] + "...",
                "Activation Frequency": n["count"],
                "First Discovered": n["first_seen"],
                "Last Active": n["last_seen"],
            }
            for n in graph["nodes"]
        ]
        st.dataframe(nodes_display, height=300)

    with g_col2:
        st.markdown(f"**Learned Temporal Transition Edges ({len(graph['edges'])} total):**")
        if graph["edges"]:
            edges_display = [
                {
                    "From Pattern": e["from"][:8] + "...",
                    "To Pattern": e["to"][:8] + "...",
                    "Transition Count": e["count"],
                    "Mean Delay (ms)": f"{e['mean_delay_ms']:.1f}",
                }
                for e in graph["edges"]
            ]
            st.dataframe(edges_display, height=300)
        else:
            st.info("No temporal transition edges formed yet. Assimilate more multi-segment speeches to form graph edges!")
else:
    st.info("No prototypes discovered yet. Click 'Assimilate Speech into Engine Memory' above to begin building the topological network.")

st.divider()

# ─── SECTION 4: MULTIMODAL SPEECH-LLM PROJECTION PAYLOAD ───────────────────────────
st.subheader("4. Multimodal Speech-LLM Reasoning Payload")
st.markdown("Inspect the exact 768-d latent payload & prosodic vector injected directly into the Speech-LLM token embedding space (Paradigm B).")

query_input = st.text_input("Enter Reasoning Query for Speech Model:", value="Analyze vocal cadence, hesitation pauses, and emotional delivery.")
if st.button("Generate Multimodal Payload"):
    if 'temp_path' in locals() and os.path.exists(temp_path):
        payload = engine.query_multimodal_speech(temp_path, prompt=query_input)
        st.success("✓ Multimodal Speech Payload Generated")
        st.json(payload)
    else:
        st.warning("Please ingest or generate an audio sample above first.")
