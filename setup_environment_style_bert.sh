set -e 

apt-get update
apt-get install ffmpeg

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper==1.0.3
pip install --force-reinstall ctranslate2==4.4.0

#-------------------------------------
# 3. 音声系ライブラリ
pip install librosa scipy pyworld pyopenjtalk num2words pydub

# -------------------------------------
# 4. Web / AI系
# -------------------------------------
pip install fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers speechbrain
