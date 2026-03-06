FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch — GPU (CUDA 12.1) if available at build time, otherwise CPU-only
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX_URL}
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY demo.py .

EXPOSE 8000

# The app auto-detects the device via torch.cuda.is_available() at runtime
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
