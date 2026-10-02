# Google Colab Remote GPU Training Guide 🚀

This directory contains notebook workflows for offloading heavy **Speech Foundation Model** training, fine-tuning, and feature extraction from your local Mac to **Google Colab GPUs** (T4 / L4 / A100), saving checkpoints directly to your **5 TB Google Drive**.

---

## 📁 Storage Architecture

* **Google Drive Base Path**: `/content/drive/MyDrive/speech_intelligence_engine/`
* **Checkpoints Path**: `/content/drive/MyDrive/speech_intelligence_engine/checkpoints/`
* **Datasets Path**: `/content/drive/MyDrive/speech_intelligence_engine/datasets/`
* **Outputs / Logs**: `/content/drive/MyDrive/speech_intelligence_engine/outputs/`

---

## 🏃 Quick Start (How to Run in Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) signed in with your account (`rakshithgoud453@gmail.com`).
2. Click **File -> Upload Notebook** and upload `speech_intelligence_colab_trainer.ipynb`.
3. Go to **Runtime -> Change runtime type** and set **Hardware accelerator** to **GPU** (T4, L4, or A100).
4. Run the notebook cells sequentially.

---

## 🛠️ What the Notebook Handles Automatically

1. **Mounts Google Drive** to access your 5 TB storage.
2. **Clones & Syncs Code**: Pulls the latest code from your Git repository.
3. **GPU Environment Setup**: Installs `torch`, `torchaudio`, `transformers`, `peft` (LoRA), `accelerate`, `bitsandbytes`, `librosa`, and `wandb`.
4. **SSL Representation Extraction**: Extracts intermediate layers (Layers 4–8 prosody, Layers 9–11 semantic) from models like `WavLM-Base-Plus` / `HuBERT`.
5. **Speech-LLM Fine-Tuning**: Trains continuous adapter projectors (e.g. Qwen2-Audio or Ultravox style) using LoRA (Low-Rank Adaptation) without consuming massive VRAM.
6. **Automatic Checkpoint Persistence**: Automatically writes model checkpoints and metrics directly into your Google Drive (`/content/drive/MyDrive/speech_intelligence_engine/checkpoints/`).
