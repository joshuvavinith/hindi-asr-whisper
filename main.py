from fastapi import FastAPI, UploadFile, File, Request
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import numpy as np
import tempfile
import os
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator

# Device configuration — use GPU if available, fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Hindi ASR API", description="Fine-tuned Whisper for Hindi speech recognition")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

MODEL_ID = "joshuavinith/whisper-small-hindi"
CHUNK_DURATION = 30  # seconds
SAMPLE_RATE = 16000

print(f"Loading model on {device}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()
print("Model loaded!")


def load_audio(file_bytes: bytes) -> np.ndarray:
    """Write bytes to a temp file, load with librosa, then clean up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio


def transcribe_chunk(audio_chunk: np.ndarray):
    """Transcribe a single audio chunk and return (transcript, confidence)."""
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    with torch.no_grad():
        result = model.generate(
            input_features,
            language="hi",
            task="transcribe",
            return_dict_in_generate=True,
            output_scores=True,
        )
    transcript = processor.batch_decode(result.sequences, skip_special_tokens=True)[0]
    confidence = None
    if hasattr(result, "sequences_scores") and result.sequences_scores is not None:
        confidence = float(torch.exp(result.sequences_scores[0]).clamp(0.0, 1.0))
    return transcript, confidence


def transcribe_audio(audio: np.ndarray):
    """Transcribe audio, chunking into 30-second segments if needed."""
    chunk_size = CHUNK_DURATION * SAMPLE_RATE
    if len(audio) <= chunk_size:
        return transcribe_chunk(audio)

    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    transcripts = []
    confidences = []
    for chunk in chunks:
        t, c = transcribe_chunk(chunk)
        transcripts.append(t)
        if c is not None:
            confidences.append(c)

    full_transcript = " ".join(transcripts).strip()
    avg_confidence = float(np.mean(confidences)) if confidences else None
    return full_transcript, avg_confidence


@app.get("/")
def root():
    return {"message": "Hindi ASR API is running!", "model": MODEL_ID, "device": device}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):
    """Transcribe a single Hindi audio file with optional confidence score."""
    audio = load_audio(await file.read())
    transcript, confidence = transcribe_audio(audio)
    response = {"transcript": transcript, "model": MODEL_ID}
    if confidence is not None:
        response["confidence"] = round(confidence, 4)
    return response


@app.post("/transcribe/stream")
@limiter.limit("10/minute")
async def transcribe_stream(request: Request, file: UploadFile = File(...)):
    """Transcribe a long audio file by splitting it into 30-second chunks."""
    audio = load_audio(await file.read())
    chunk_size = CHUNK_DURATION * SAMPLE_RATE
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        t, c = transcribe_chunk(chunk)
        chunk_result = {
            "chunk": i + 1,
            "transcript": t,
            "start_time": round(i * CHUNK_DURATION, 2),
            "end_time": round(min((i + 1) * CHUNK_DURATION, len(audio) / SAMPLE_RATE), 2),
        }
        if c is not None:
            chunk_result["confidence"] = round(c, 4)
        results.append(chunk_result)

    full_transcript = " ".join(r["transcript"] for r in results).strip()
    return {"transcript": full_transcript, "chunks": results, "model": MODEL_ID}


@app.post("/transcribe/batch")
@limiter.limit("10/minute")
async def transcribe_batch(request: Request, files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files in one request."""
    results = []
    for file in files:
        audio = load_audio(await file.read())
        transcript, confidence = transcribe_audio(audio)
        result = {"filename": file.filename, "transcript": transcript}
        if confidence is not None:
            result["confidence"] = round(confidence, 4)
        results.append(result)
    return {"results": results, "model": MODEL_ID}


@app.post("/detect-language")
@limiter.limit("10/minute")
async def detect_language(request: Request, file: UploadFile = File(...)):
    """Detect the language of the audio using Whisper's built-in language detection."""
    audio = load_audio(await file.read())
    # Use only the first 30 seconds for language detection
    audio_clip = audio[:CHUNK_DURATION * SAMPLE_RATE]
    inputs = processor(audio_clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        # Generate without forcing language so Whisper predicts the language token
        result = model.generate(
            input_features,
            return_dict_in_generate=True,
            max_new_tokens=5,
        )

    # The language token is one of the first tokens after <|startoftranscript|>
    token_ids = result.sequences[0].tolist()
    detected_lang = "unknown"
    for token_id in token_ids[1:4]:
        token_str = processor.tokenizer.decode([token_id])
        # Language tokens look like <|hi|>, <|en|>, etc.
        if token_str.startswith("<|") and token_str.endswith("|>"):
            inner = token_str[2:-2]
            if 2 <= len(inner) <= 4 and inner.isalpha():
                detected_lang = inner
                break

    response = {"detected_language": detected_lang, "model": MODEL_ID}
    if detected_lang not in ("hi", "unknown"):
        response["warning"] = (
            f"Audio appears to be in '{detected_lang}', not Hindi ('hi'). "
            "Transcription may be less accurate."
        )
    return response
