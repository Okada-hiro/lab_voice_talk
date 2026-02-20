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

#-------------------------------------
# 3. 音声系ライブラリ
pip install librosa scipy soundfile pyworld pyopenjtalk num2words pydub

# -------------------------------------
# 4. Web / AI系
# -------------------------------------
pip install fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers speechbrain

# -------------------------------------
# 5. Qwen3-TTS (streaming fork)
#    - Install streaming-enabled qwen_tts as active package
# -------------------------------------
WORKDIR="${WORKDIR:-/workspace}"
QWEN3_TTS_STREAMING_REPO="${QWEN3_TTS_STREAMING_REPO:-https://github.com/rekuenkdr/Qwen3-TTS-streaming.git}"
mkdir -p "${WORKDIR}"
if [ ! -d "${WORKDIR}/Qwen3-TTS-streaming" ]; then
  git clone "${QWEN3_TTS_STREAMING_REPO}" "${WORKDIR}/Qwen3-TTS-streaming"
fi

pip uninstall -y qwen-tts qwen_tts || true
pip install -e "${WORKDIR}/Qwen3-TTS-streaming"
