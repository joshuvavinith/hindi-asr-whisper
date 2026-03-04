# 🎙️ Hindi ASR — Fine-tuned Whisper

> End-to-end Hindi Automatic Speech Recognition pipeline: dataset engineering, Whisper-small fine-tuning, disfluency detection, and lattice-based evaluation.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-joshuavinith%2Fwhisper--small--hindi-blue)](https://huggingface.co/joshuavinith/whisper-small-hindi)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-green)](https://huggingface.co/spaces/joshuavinith/hindi-asr-demo)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds a complete Hindi ASR system from scratch — from raw conversational recordings to a deployed REST API. It covers four research and engineering stages:

1. **Dataset Engineering** — Constructing a high-quality utterance-level corpus from 104 long-form conversational recordings (~10 hrs)
2. **Model Fine-Tuning** — Adapting Whisper-small to conversational Hindi using HuggingFace Seq2SeqTrainer
3. **Disfluency Detection** — Rule-based pipeline for detecting fillers, repetitions, and prolongations in Hindi speech
4. **Lattice Evaluation** — Novel multi-system consensus framework for fairer WER computation

**Result:** WER reduced from **48.26% → 31.51%** on conversational Hindi.

---

## Architecture

```
Raw Audio (.wav) ──► Preprocessing Pipeline ──► Fine-tuned Whisper-small ──► Hindi Transcript
                          │
                          ├── URL reconstruction
                          ├── JSON-aligned segmentation (pydub)
                          ├── 16kHz mono standardization (librosa)
                          └── Log-mel spectrogram extraction
```

```
FastAPI App
    │
    ├── POST /transcribe  ──► load audio ──► processor ──► model.generate() ──► transcript
    ├── GET  /health
    └── GET  /docs        ──► interactive Swagger UI
```

---

## Results

| Model | WER (%) |
|-------|---------|
| Whisper-small (pretrained baseline) | 48.26 |
| Fine-tuned — Epoch 1 | 39.24 |
| Fine-tuned — Epoch 2 | 33.01 |
| **Fine-tuned — Epoch 3 (best)** | **31.51** |
| Fine-tuned — Epoch 4 | 32.48 |

The fine-tuned model also outperformed the pretrained baseline on an **external clean read-speech Hindi benchmark**, confirming cross-domain generalization.

---

## Live Demo

Try it instantly in your browser — no setup needed:

🔗 **[huggingface.co/spaces/joshuavinith/hindi-asr-demo](https://huggingface.co/spaces/joshuavinith/hindi-asr-demo)**

Upload or record Hindi audio and get a transcript in real time.

---

## Quickstart

### Option 1 — Python (direct inference)

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import torch

processor = WhisperProcessor.from_pretrained("joshuavinith/whisper-small-hindi")
model = WhisperForConditionalGeneration.from_pretrained("joshuavinith/whisper-small-hindi")
model.eval()

audio, sr = librosa.load("your_audio.wav", sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"], language="hi", task="transcribe")

print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

### Option 2 — REST API (Docker)

```bash
# Clone and build
git clone https://github.com/joshuvavinith/hindi-asr-whisper
cd hindi-asr-whisper
docker build -t hindi-asr-api .

# Run
docker run -p 8000:8000 hindi-asr-api
```

API is now live at `http://localhost:8000`

**Interactive docs:** `http://localhost:8000/docs`

**Transcribe via curl:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -F "file=@your_hindi_audio.wav"
```

**Response:**
```json
{
  "transcript": "आपका स्वागत है",
  "model": "joshuavinith/whisper-small-hindi"
}
```

### Option 3 — Run API without Docker

```bash
pip install fastapi uvicorn transformers torch librosa python-multipart huggingface_hub
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
hindi-asr-whisper/
├── main.py                          # FastAPI app
├── Dockerfile                       # Docker container definition
├── requirements.txt                 # Python dependencies
├── notebooks/
│   ├── final_josh_preprocessing.ipynb   # Data pipeline
│   ├── Finetune_model_1.ipynb           # Fine-tuning run 1
│   └── Finetune_model_2_new.ipynb       # Fine-tuning run 2 (best)
└── README.md
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-small |
| Dataset | ~10 hrs conversational Hindi (5,732 utterances) |
| Epochs | 4 (best checkpoint: epoch 3) |
| Learning rate | 1e-5 |
| Optimizer | AdamW + warmup (500 steps) |
| Mixed precision | FP16 |
| Framework | HuggingFace Seq2SeqTrainer |

---

## Research Contributions

- **Custom dataset pipeline** — Programmatic reconstruction of 104 conversational Hindi recordings with JSON-aligned segmentation
- **Disfluency detection** — Rule-based Hindi disfluency pipeline (fillers, repetitions, prolongations) producing a labeled clip-level dataset
- **Spelling analysis** — 65.5% of 7,457 unique transcript tokens classified as orthographically noisy using Devanagari script validation
- **Lattice consensus evaluation** — Novel framework aligning 6 ASR hypotheses via Levenshtein dynamic programming for fairer WER computation

📄 Full technical report (18 pages) — ArXiv preprint in preparation.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/transcribe` | Transcribe Hindi audio file |
| GET | `/docs` | Interactive Swagger UI |

---

## Requirements

- Python 3.9+
- Docker (for containerized deployment)
- CPU inference supported (no GPU required)

---

## Limitations

- Optimized for conversational Hindi; may underperform on formal/domain-specific speech
- Whisper-small (244M params) — larger variants would yield lower WER
- No speaker diarization support

---

## Author

**Joshuva Vinith**  
B.Tech — Artificial Intelligence & Data Science

📧 joshuvinith@gmail.com  
🔗 [HuggingFace](https://huggingface.co/joshuavinith) | [LinkedIn](https://linkedin.com/in/joshuva-vinith)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
