#!/bin/bash

set -e  # エラーが起きたら停止

# -------------------------------------
# 1. システムライブラリ
# -------------------------------------
apt-get update
apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 ffmpeg
# cuDNN ライブラリをシステムに認識させる
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ldconfig

# -------------------------------------
# 2. PyTorch 
# -------------------------------------
REQUIRED_TORCH_VERSION="2.8.0+cu128"

INSTALLED_TORCH_VERSION=$(pip show torch | grep Version | awk '{print $2}')

if [ "$INSTALLED_TORCH_VERSION" != "$REQUIRED_TORCH_VERSION" ]; then
    echo "Installing PyTorch $REQUIRED_TORCH_VERSION ..."
    pip install --force-reinstall torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
else
    echo "PyTorch $REQUIRED_TORCH_VERSION is already installed. Skipping."
fi


#-------------------------------------
# 3. 音声系ライブラリ
pip install librosa scipy pyworld pyopenjtalk num2words pydub

# -------------------------------------
# 4. Web / AI系
# -------------------------------------
pip install fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers faster-whisper

