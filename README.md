# 🧠 NeuralLens — Unified AI Studio

<div align="center">

![NeuralLens Banner](https://img.shields.io/badge/NeuralLens-AI%20Studio-22d3ee?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Server-black?style=for-the-badge&logo=flask)
![GPU](https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia)

**A production-grade, multi-model AI platform combining Image Captioning, Code Generation, Text Generation, and Semantic Search — all served from a single unified server.**

[🚀 Live Demo](#quick-start) · [📖 API Docs](#api-reference) · [🤖 AI Studio](#ai-studio)

</div>

---

## 🌟 What is NeuralLens?

NeuralLens is a **full-stack AI inference platform** built from scratch, unifying four custom-trained deep learning models into one seamless, GPU-accelerated server. It features a premium dark-mode web interface with real-time telemetry, cross-model pipelines, and a REST API.

```
┌─────────────────────────────────────────────────────────┐
│              NeuralLens Unified Server                   │
│                   localhost:5055                         │
├──────────────┬─────────────┬────────────┬───────────────┤
│  Image       │  KhemetCode │  MiniGPT   │  BERT Semantic│
│  Captioning  │  NL → Code  │  Text Gen  │  Search       │
│  ViT + GPT-2 │  Transformer│  GPT arch  │  Siamese BERT │
│  ~340M params│  custom tok │  256-dim   │  12.6M params │
└──────────────┴─────────────┴────────────┴───────────────┘
```

---

## 🎯 Models & Capabilities

### 🖼️ Image Captioning — ViT + GPT-2 Decoder
- **Architecture:** Vision Transformer (ViT) encoder + GPT-2 style autoregressive decoder
- **Parameters:** ~340M
- **Input:** Any image (224×224 resize, RGB normalization)
- **Output:** Natural language caption (up to 39 tokens)
- **Features:** Temperature sampling, Beam Search (1–5), style presets (Descriptive / Concise / Creative)
- **Training:** Fine-tuned on COCO Captions with incremental learning support
- **Device:** GPU-accelerated (`/physical_device:GPU:0`)

### 💻 KhemetCode — Natural Language → Code Transformer
- **Architecture:** Encoder–Decoder Transformer with custom SentencePiece tokenizers
- **Task:** Translates natural language descriptions into executable code
- **Languages:** Python, JavaScript, C, C++, Java
- **Tokenization:** Separate NL tokenizer (`nl_tokenizer.model`) and code tokenizer (`code_tokenizer.model`)
- **Features:** Temperature control, top-k sampling, character-count metrics

### ✍️ MiniGPT — Text Generation
- **Architecture:** Custom GPT (6-layer, 4-head, 256-dim embeddings)
- **Vocabulary:** Custom BPE tokenizer (~50K tokens)
- **Task:** Autoregressive text continuation from a prompt
- **Features:** Max token control (20–400), temperature sampling, background training

### 🔍 Semantic Search — BERT Siamese Network
- **Architecture:** Siamese BERT-based encoder with contrastive loss
- **Parameters:** 12.65M
- **Task:** Measures semantic similarity between two sentences (0–100%)
- **Output:** Similarity score + human-readable interpretation label
- **Training:** Continual learning — never forgets previous training pairs

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow flask flask-cors numpy pillow psutil h5py sentencepiece transformers tokenizers
```

### Run the Unified Server
```bash
cd "ViT and GPT"
python3 app.py
```

The server starts all 4 models automatically:
```
============================================================
🚀 VisionMind Alpha 1.0.0
============================================================
[1/4] Loading tokenizer…
[2/4] Loading model (1.3GB)…  ✓ 264 tensors
[3/4] Detecting GPU…
[4/4] System check…
============================================================
✓ Processing Device: /physical_device:GPU:0
✓ AI Studio:      http://localhost:5055/studio
✓ Caption API:    http://localhost:5055/api/caption
============================================================
[5/5] Loading Studio models (KhemetCode + MiniGPT + BERT)…
  ↳ Loading in background — status dots go green when ready
  ✓ KhemetCode loaded
  ✓ MiniGPT Text Gen loaded
  ✓ Semantic Search (BERT) loaded
```

### Access the UI
| Interface | URL |
|---|---|
| **AI Studio** (all models) | http://localhost:5055/studio |
| Original NeuralLens UI | http://localhost:5055 |

---

## 🤖 AI Studio

The AI Studio is a unified dashboard that connects all 4 models with a live sidebar showing each model's status:

| Tab | Model | Features |
|---|---|---|
| 🖼️ **Image Captioning** | ViT+GPT-2 | Upload image, style presets, typewriter animation, caption history |
| 💻 **KhemetCode** | NL→Code Transformer | Describe code in English, get working code with syntax highlighting |
| ✍️ **Text Generation** | MiniGPT | Prompt continuation with token/temperature sliders |
| 🔍 **Semantic Search** | Siamese BERT | Compare two sentences with animated similarity bar |
| ⚡ **AI Workflow** | All models | Chain outputs between models (Caption → Code, Caption → Text, etc.) |
| 📊 **Dashboard** | System | Live CPU/RAM/GPU meters, all service statuses |

---

## 📡 API Reference

All endpoints on `http://localhost:5055`

### Image Captioning
```bash
POST /api/caption
Content-Type: multipart/form-data

curl -X POST http://localhost:5055/api/caption \
  -F "image=@photo.jpg" \
  -F "temperature=1.0" \
  -F "beam=1"
```
**Response:**
```json
{
  "caption": "a dog playing in a sunny park",
  "time_sec": 0.42,
  "device": "/physical_device:GPU:0"
}
```

### Code Generation (KhemetCode)
```bash
POST /api/studio/code
Content-Type: application/json

curl -X POST http://localhost:5055/api/studio/code \
  -H "Content-Type: application/json" \
  -d '{"text": "sort a list of dictionaries by key", "max_len": 200}'
```

### Text Generation (MiniGPT)
```bash
POST /api/studio/text
Content-Type: application/json

curl -X POST http://localhost:5055/api/studio/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 100, "temperature": 0.8}'
```

### Semantic Similarity (BERT)
```bash
POST /api/studio/search
Content-Type: application/json

curl -X POST http://localhost:5055/api/studio/search \
  -H "Content-Type: application/json" \
  -d '{"sentence1": "The cat sat on the mat", "sentence2": "A feline rested on the rug"}'
```
**Response:**
```json
{
  "similarity": 0.87,
  "percent": 87.0,
  "label": "Very Similar",
  "time_sec": 0.031
}
```

### System Status
```bash
GET /api/studio/status    # All model statuses
GET /api/telemetry        # CPU, RAM, GPU live stats
GET /api/stats            # Inference counts, history
```

---

## 🏗️ Project Structure

```
ViT and GPT/
├── app.py                          # Unified Flask server (all 4 models)
├── index.html                      # NeuralLens original UI
├── unified_studio.html             # AI Studio unified UI
├── epoch_03_valloss_0.0626.keras   # ViT+GPT-2 model weights (1.3GB)
├── caption_tokenizer-vocab.json    # BPE vocabulary
├── caption_tokenizer-merges.txt    # BPE merge rules
├── neurallens_db.json              # Persistent storage
└── training_samples/               # Fine-tuning image cache

~/files_for_model/                  # KhemetCode
├── app.py                          # generate_code() function
├── best_transformer.keras          # Transformer weights (~155MB)
├── nl_tokenizer.model              # SentencePiece NL tokenizer
└── code_tokenizer.model            # SentencePiece code tokenizer

~/Text_Generation/                  # MiniGPT
├── model_utils.py                  # generate_text() function
├── minigpt.keras                   # MiniGPT weights (~100MB)
└── minigpt_tokenizer.json          # BPE tokenizer

~/Semantic_Search_Engine/           # BERT Semantic Search
├── model_utils.py                  # predict_similarity() function
└── semantic_bert.keras             # Siamese BERT weights (~145MB)
```

---

## ⚙️ Technical Highlights

- **GPU-aware multi-model loading** — All 4 models share one TF session; GPU device conflicts resolved via context patching
- **Background threading** — Studio models load in daemon threads; server responds immediately
- **Incremental fine-tuning** — Image captioning model supports on-device fine-tuning with configurable epochs (1–20)
- **Cross-model workflows** — Caption output automatically feeds into Code Gen, Text Gen, or Semantic Search
- **Real-time telemetry** — CPU/RAM/GPU polled every 2s and visualized in the dashboard
- **API key management** — Token-based auth with `secrets.token_urlsafe(32)` for production endpoints
- **Threaded Flask** — `threaded=True` handles concurrent inference requests without blocking

---

## 📊 Model Performance

| Model | Inference Time | Memory | Device |
|---|---|---|---|
| Image Captioning (ViT+GPT-2) | ~0.4s | 1.3 GB | GPU |
| KhemetCode (NL→Code) | ~0.2s | 155 MB | CPU/GPU |
| MiniGPT Text Gen | ~0.3s | 100 MB | CPU |
| BERT Semantic Search | ~0.03s | 145 MB | CPU |

---

## 🔬 Training Details

### Image Captioning
- **Dataset:** COCO Captions (fine-tune split)
- **Optimizer:** Adam, lr=1e-5
- **Loss:** Masked Sparse Categorical Crossentropy
- **Validation Loss:** 0.0626 (epoch 3)
- **Incremental training:** Supports adding new image-caption pairs via the web UI

### Semantic Search
- **Loss:** Contrastive Loss
- **Continual learning:** New training pairs are saved and never forgotten
- **7+ training samples** retained across restarts

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.

---

<div align="center">

**Built with ❤️ using TensorFlow, Flask, and custom deep learning architectures**

⭐ Star this repo if you found it useful!

</div>
