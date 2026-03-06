"""Unit tests for the Hindi ASR FastAPI application.

The Whisper model and processor are mocked at import time so that no actual
model weights are downloaded or used during testing.
"""

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_wav_bytes() -> bytes:
    """Return bytes of a minimal valid mono 16-bit PCM WAV file (0.5 s silence)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock model / processor — must be set up BEFORE importing main
# ---------------------------------------------------------------------------

_mock_proc = MagicMock()
_mock_model = MagicMock()

# processor(audio, ...) → {"input_features": <tensor>}
_mock_proc.return_value = {"input_features": torch.zeros(1, 80, 3000)}

# processor.batch_decode(...) → ["नमस्ते"]
_mock_proc.batch_decode.return_value = ["नमस्ते"]

# processor.tokenizer.decode([token_id]) → "<|hi|>"
_mock_proc.tokenizer.decode.return_value = "<|hi|>"

# model.generate(...) returns an object with .sequences and .sequences_scores
_mock_result = MagicMock()
_mock_result.sequences = torch.zeros((1, 10), dtype=torch.long)
_mock_result.sequences_scores = torch.tensor([-0.5])
_mock_model.generate.return_value = _mock_result

# model.to(device) → model itself
_mock_model.to.return_value = _mock_model
_mock_model.eval.return_value = _mock_model

with (
    patch(
        "transformers.WhisperProcessor.from_pretrained",
        return_value=_mock_proc,
    ),
    patch(
        "transformers.WhisperForConditionalGeneration.from_pretrained",
        return_value=_mock_model,
    ),
):
    from main import app  # noqa: E402  (import after patching)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_librosa_load():
    """Replace librosa.load so no real audio decoding occurs."""
    with patch(
        "librosa.load",
        return_value=(np.zeros(8000, dtype=np.float32), 16000),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root():
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hindi ASR API is running!"
    assert "model" in data


@pytest.mark.asyncio
async def test_health():
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_transcribe():
    from httpx import ASGITransport, AsyncClient

    wav_bytes = create_test_wav_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "model" in data


@pytest.mark.asyncio
async def test_transcribe_stream():
    from httpx import ASGITransport, AsyncClient

    wav_bytes = create_test_wav_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/transcribe/stream",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) >= 1


@pytest.mark.asyncio
async def test_detect_language():
    from httpx import ASGITransport, AsyncClient

    wav_bytes = create_test_wav_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/detect-language",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "detected_language" in data
    assert "model" in data


@pytest.mark.asyncio
async def test_transcribe_batch():
    from httpx import ASGITransport, AsyncClient

    wav_bytes = create_test_wav_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/transcribe/batch",
            files=[
                ("files", ("test1.wav", wav_bytes, "audio/wav")),
                ("files", ("test2.wav", wav_bytes, "audio/wav")),
            ],
        )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    for item in data["results"]:
        assert "filename" in item
        assert "transcript" in item
