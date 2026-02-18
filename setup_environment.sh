set -e 

apt-get update
apt-get install -y ffmpeg

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
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
