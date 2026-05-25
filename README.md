<div align="center">
  <img src="https://raw.githubusercontent.com/Alouakhalid/NeuralLens/main/assets/banner.png" alt="NeuralLens Banner" width="100%">

  # 🧠 NeuralLens: Continual Learning Image Captioning
  **An ultra-premium, real-time Vision-Transformer (ViT) & GPT-2 decoder platform with a stunning web GUI.**
  
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
</div>

<br>

## 🚀 Overview

**NeuralLens** is an advanced AI platform designed for seamless **Image Captioning** utilizing a custom hybrid architecture. We have combined the spatial attention mechanisms of a **Vision Transformer (ViT)** encoder with the powerful, autoregressive sequential decoding of a **GPT-2** style model. 

Built entirely in pure Python (TensorFlow/Keras + Flask), NeuralLens provides an ultra-premium, "glassmorphism" web GUI out-of-the-box, allowing users to interact with the models, monitor system hardware telemetry, and uniquely **fine-tune the model directly from the browser.**

---

## ✨ Premium Features

*   🖼️ **VisionMind Core**: High-accuracy ViT + GPT-2 Image Captioning capable of semantic scene understanding.
*   🧠 **Continual Learning Pipeline**: A built-in "Training Studio" that allows users to drag-and-drop new unseen images, provide ground-truth captions, and perform targeted backpropagation (epoch training) directly via the UI without overwriting existing neural weights.
*   📊 **Real-Time Telemetry Island**: Live monitoring of CPU, RAM, and GPU memory allocation using `psutil` and native NVIDIA metrics.
*   🎨 **Spectacular UI/UX**: Designed without bloated JavaScript frameworks. The platform relies on pure HTML5 Canvas for dynamic particle backgrounds, modern CSS glassmorphism, and smooth cubic-bezier transitions.
*   🔑 **Admin API Key Management**: Generate scoped API keys with expiration dates to securely expose your model inference endpoints to external applications.

---

## 🛠️ Architecture Deep Dive

The neural network is defined in Keras and features:
1.  **ViT Encoder**: Extracts 16x16 patch embeddings from a $224 \times 224$ image, passing them through multi-head self-attention layers to capture spatial semantics.
2.  **GPT-2 Decoder**: Ingests the encoded image representations via cross-attention. It uses a custom BPE (Byte-Pair Encoding) tokenizer and generates textual descriptions autoregressively.
3.  **Unified Flask Server**: A highly optimized multithreaded server acting as both the REST API provider and the web application host on port `5055`.

---

## 💻 Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Alouakhalid/NeuralLens.git
   cd NeuralLens
   ```

2. **Install Dependencies**
   ```bash
   pip install tensorflow flask flask-cors pillow psutil h5py numpy
   ```

3. **Provide Pre-trained Weights**
   Place your Keras weights file (e.g., `epoch_03_valloss_0.0626.keras`) and tokenizer JSON into the root directory. *(Weights are not included in the repo due to GitHub LFS constraints).*

4. **Launch the Studio**
   ```bash
   python app.py
   ```
   Navigate your browser to: **[http://localhost:5055](http://localhost:5055)**

---

## 🔌 API Documentation

NeuralLens comes with a fully featured programmatic REST API. 

### Generate a Caption (Inference)
```bash
curl -X POST http://localhost:5055/api/v1/caption \
  -H "X-API-Key: nlk-your-secure-api-key" \
  -F "image=@/path/to/photo.jpg" \
  -F "temperature=1.0" \
  -F "beam=1"
```

### Upload a Training Sample
```bash
curl -X POST http://localhost:5055/api/train/upload \
  -F "image=@/path/to/new_data.jpg" \
  -F "caption=A realistic description of the new image"
```

---

<div align="center">
  <b>Developed by Khalid Ali</b><br>
  AI Engineer & Deep Learning Researcher
</div>
