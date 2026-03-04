from fastapi import FastAPI, UploadFile, File
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import numpy as np
import tempfile
import os

app = FastAPI(title="Hindi ASR API", description="Fine-tuned Whisper for Hindi speech recognition")

MODEL_ID = "joshuavinith/whisper-small-hindi"

print("Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded!")

@app.get("/")
def root():
    return {"message": "Hindi ASR API is running!", "model": MODEL_ID}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    os.unlink(tmp_path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            language="hi",
            task="transcribe"
        )
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return {"transcript": transcript, "model": MODEL_ID}
