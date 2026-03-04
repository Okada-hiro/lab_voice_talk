#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y ffmpeg sox git build-essential ninja-build

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${PY_VER}" != "3.10" ]; then
  echo "[WARN] Recommended Python is 3.10, current: ${PY_VER}"
fi

pip install -U pip setuptools wheel packaging psutil ninja
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
MAX_JOBS=4 pip install -U flash-attn==2.8.3 --no-build-isolation

pip uninstall -y ctranslate2 faster-whisper || true
pip install -U ctranslate2 faster-whisper

# -------------------------------------
# 3. 音声系ライブラリ
# -------------------------------------
pip install -U librosa scipy soundfile pyworld pyopenjtalk num2words pydub

# -------------------------------------
# 4. Web / AI系
# -------------------------------------
pip install -U fastapi uvicorn[standard] google-generativeai huggingface_hub loguru transformers speechbrain



# 0) そもそもworkspace/lab_voice_talkだが、workspace/faster-qwen3ttsにする

# 1) 競合しやすいものを入れ直し
pip uninstall -y qwen-tts faster-qwen3-tts transformers huggingface_hub speechbrain

# 2) 互換が取りやすい組み合わせで再インストール
pip install "transformers==4.57.3" "qwen-tts==0.1.1"
pip install "huggingface_hub<1.0" "speechbrain>=1.0.0"

# 3) faster-qwen3-tts はローカルを優先（推奨）
pip install -e /workspace/faster-qwen3-tts

