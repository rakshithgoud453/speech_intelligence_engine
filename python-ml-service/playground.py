import sys
import os
import streamlit as st
import numpy as np
import torch

# Add repository root to python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from speech_intelligence_engine import SpeechLearningEngine

st.set_page_config(page_title="Speech Intelligence Engine Explorer", layout="wide")

st.title("🎙️ Speech Intelligence Engine Explorer")
st.markdown(
    "A self-supervised, native speech representation and learning system. "
    "Understands vocal prosody, pitch, hesitation, and acoustic latents directly."
)

# Sidebar Configuration
st.sidebar.header("Engine Architecture")
engine_mode = st.sidebar.radio(
    "Select System Paradigm:",
    [
        "🧠 Paradigm B: Native Speech Intelligence Engine (SSL & Multimodal)",
        "📝 Paradigm A: Cascaded ASR & Diarization (Legacy Tooling)",
    ],
)

@st.cache_resource
def get_native_engine():
    return SpeechLearningEngine(model_type="hubert", replay_capacity=500, max_prototypes=100)

if "Paradigm B" in engine_mode:
    st.subheader("🧠 Native Speech Representation & Multimodal Reasoning Engine")
    st.markdown("Ingests raw audio (16kHz), extracts 12-layer HuBERT/WavLM hidden representations, and builds evolving memory structures.")

    engine = get_native_engine()

    uploaded_file = st.file_uploader("Upload Speech Recording (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Assimilate into Speech Engine"):
                with st.spinner("Assimilating raw audio into memory..."):
                    res = engine.assimilate(temp_path, recording_id=uploaded_file.name)
                    st.success(f"✓ Audio Assimilated Successfully ({res.duration_sec:.2f}s)")
                    st.json({
                        "duration_sec": res.duration_sec,
                        "voiced_segments": res.num_segments,
                        "episodes_stored": res.num_episodes,
                        "novel_structures": res.num_novel,
                        "novelty_rate": f"{res.novelty_rate:.1%}",
                    })

        with col2:
            query_prompt = st.text_input(
                "Multimodal Speech Query:",
                value="Analyze vocal inflection, tempo, hesitation, and emotional nuance."
            )
            if st.button("Query Multimodal Engine"):
                with st.spinner("Extracting layer-wise latents & generating multimodal prompt..."):
                    payload = engine.query_multimodal_speech(temp_path, prompt=query_prompt)
                    st.success("✓ Multimodal Speech Representation Payload Created")
                    st.json(payload)

        st.divider()
        st.subheader("📊 Engine Memory & Prototype Graph Snapshot")
        state = engine.get_state()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Episodes Learned", state.total_episodes)
        m2.metric("Acoustic Prototypes", state.total_prototypes)
        m3.metric("Relational Transitions", state.total_transitions)
        m4.metric("Replay Buffer Size", f"{state.replay_buffer_size} / {state.replay_buffer_total_seen}")

        graph = engine.get_prototype_graph()
        if graph["nodes"]:
            st.markdown(f"**Discovered Unsupervised Speech Prototypes ({len(graph['nodes'])} Nodes):**")
            st.dataframe(graph["nodes"])

else:
    st.subheader("📝 Legacy Paradigm A: Cascaded ASR & Diarization")
    st.info("Cascaded tools (faster-whisper + pyannote) process raw audio into flat text.")

    try:
        from pyannote.audio import Pipeline
        from faster_whisper import WhisperModel
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False

    if not ML_AVAILABLE:
        st.error("Cascaded ML dependencies (faster_whisper / pyannote) missing in this environment.")
    else:
        uploaded_file = st.file_uploader("Upload Audio for Transcription", type=["wav", "mp3"], key="legacy_upload")
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            temp_path = f"/tmp/legacy_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Run ASR Transcription"):
                with st.spinner("Transcribing..."):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = WhisperModel("small", device=device, compute_type="int8")
                    segments, info = model.transcribe(temp_path, beam_size=5)
                    for seg in list(segments):
                        st.markdown(f"**[{seg.start:.2f}s - {seg.end:.2f}s]:** {seg.text}")
