#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg sox git build-essential libsndfile1 libportaudio2 portaudio19-dev

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[INFO] Python version: ${PY_VER}"

pip install -U pip setuptools wheel

# Torch stack (required by main.py / speaker filter / VAD)
if ! python -c 'import torch' >/dev/null 2>&1; then
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
  echo "[INFO] torch already installed. skip torch install"
fi

# ASR / audio
pip install -U numpy scipy soundfile librosa
pip install -U ctranslate2 faster-whisper

# Speaker verification
pip install -U speechbrain

# App server / env
pip install -U fastapi "uvicorn[standard]" websockets python-dotenv loguru

# LLM + TTS API clients
pip install -U openai google-generativeai

echo "[INFO] OpenAI pipeline environment setup completed."
echo "[INFO] Set API keys before running:"
echo "       export OPENAI_API_KEY='your_api_key'"
echo "       export GOOGLE_API_KEY='your_google_api_key'"
