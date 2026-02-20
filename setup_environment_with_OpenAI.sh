#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg sox git build-essential ninja-build libsndfile1 libportaudio2 portaudio19-dev

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${PY_VER}" != "3.10" ]; then
  echo "[WARN] Recommended Python is 3.10, current: ${PY_VER}"
fi

pip install -U pip setuptools wheel packaging psutil ninja

# Torch stack (same base as setup_environment.sh)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# ASR dependencies
pip uninstall -y ctranslate2 faster-whisper || true
pip install -U ctranslate2 faster-whisper

# Audio libraries (include librosa for transcribe_func.py)
pip install -U librosa scipy soundfile pyworld pyopenjtalk num2words pydub

# Web / AI libraries for main.py pipeline
pip install -U fastapi "uvicorn[standard]" google-generativeai openai loguru python-dotenv websockets

# Keep SpeechBrain and HF Hub compatible
pip install -U "huggingface_hub==0.25.2"
pip install -U "speechbrain==0.5.16"

echo "[INFO] OpenAI pipeline environment setup completed."
echo "[INFO] Set API keys before running:"
echo "       export OPENAI_API_KEY='your_api_key'"
echo "       export GOOGLE_API_KEY='your_google_api_key'"
