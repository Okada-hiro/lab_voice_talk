#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg sox git libportaudio2 portaudio19-dev

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[INFO] Python version: ${PY_VER}"

pip install -U pip setuptools wheel

# OpenAI TTS + audio streaming/playback
pip install -U openai numpy sounddevice soundfile scipy

# Optional: if you run the websocket/web app too
pip install -U fastapi "uvicorn[standard]" websockets python-dotenv loguru

echo "[INFO] OpenAI environment setup completed."
