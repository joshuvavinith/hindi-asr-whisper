"""Gradio demo for the Hindi ASR fine-tuned Whisper model.

Run with:
    python demo.py
"""

import gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import numpy as np

MODEL_ID = "joshuavinith/whisper-small-hindi"
CHUNK_DURATION = 30  # seconds
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()
print("Model loaded!")


def transcribe(audio_path: str) -> str:
    """Transcribe a Hindi audio file given its file path."""
    if audio_path is None:
        return "Please upload an audio file."

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_size = CHUNK_DURATION * SAMPLE_RATE
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    transcripts = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"].to(device),
                language="hi",
                task="transcribe",
            )
        transcripts.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )

    return " ".join(transcripts).strip()


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Hindi Audio (.wav or .mp3)"),
    outputs=gr.Textbox(label="Transcription"),
    title="🎙️ Hindi ASR — Fine-tuned Whisper",
    description=(
        "Upload a Hindi audio file (.wav or .mp3) to get the transcript "
        "using a fine-tuned Whisper-small model. "
        "Long files are automatically split into 30-second chunks."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
