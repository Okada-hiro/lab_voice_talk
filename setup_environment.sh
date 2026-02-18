set -e 

apt-get update
apt-get install -y ffmpeg sox git build-essential ninja-build

PY_VER="$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
if [ "${PY_VER}" != "3.10" ]; then
  echo "[WARN] Recommended Python is 3.10, current: ${PY_VER}"
fi

pip install -U pip setuptools wheel packaging psutil ninja
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
MAX_JOBS=4 pip install -U flash-attn==2.8.3 --no-build-isolation

pip install faster-whisper==1.0.3
pip install --force-reinstall ctranslate2==4.4.0

#-------------------------------------
# 3. 音声系ライブラリ
pip install librosa scipy soundfile pyworld pyopenjtalk num2words pydub

# -------------------------------------
# 4. Web / AI系
# -------------------------------------
pip install fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers speechbrain

# -------------------------------------
# 5. Qwen3-TTS (standard install + source clone)
#    - Standard pip package install (no manual steps required)
#    - Clone source repo for reference / optional manual patching later
# -------------------------------------
pip install qwen-tts

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "${WORKDIR}"
if [ ! -d "${WORKDIR}/Qwen3-TTS" ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git "${WORKDIR}/Qwen3-TTS"
fi
