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
pip install -U fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers speechbrain

# -------------------------------------
# 5. Qwen3-TTS (base: setup_environment_qwen3tts.sh と同等)
# -------------------------------------
pip install -U qwen-tts

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "${WORKDIR}"
if [ ! -d "${WORKDIR}/Qwen3-TTS" ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git "${WORKDIR}/Qwen3-TTS"
fi

# -------------------------------------
# 6. Qwen3-TTS streaming fork を追加導入（上位互換ポイント）
# -------------------------------------
QWEN3_TTS_STREAMING_REPO="${QWEN3_TTS_STREAMING_REPO:-https://github.com/dffdeeq/Qwen3-TTS-streaming.git}"
if [ ! -d "${WORKDIR}/Qwen3-TTS-streaming" ]; then
  git clone "${QWEN3_TTS_STREAMING_REPO}" "${WORKDIR}/Qwen3-TTS-streaming"
fi

# streaming fork を有効化
pip uninstall -y qwen-tts qwen_tts || true
pip install -e "${WORKDIR}/Qwen3-TTS-streaming"

# 動作確認（stream API の有無）
python - <<'PY'
try:
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda")
    print("[CHECK] has_stream_generate_voice_clone:", hasattr(m, "stream_generate_voice_clone"))
except Exception as e:
    print("[CHECK] streaming check skipped/failed:", e)
PY

