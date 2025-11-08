#!/bin/bash
# /workspace/setup_environment.sh
# 音声チャットボット環境構築スクリプト（リポジトリ自動取得・更新版）

# スクリプトのいずれかのコマンドが失敗したら停止
set -e

# --- リポジトリ設定 ---
# 1. whisper-streaming の設定
# (transcribe_func.py が ufal/whisper_streaming の whisper_online.py を使うと想定)
WHISPER_REPO_URL="https://github.com/ufal/whisper_streaming.git"
WHISPER_DIR="/workspace/whisper-streaming"

# 2. fish-speech の設定
FISH_REPO_URL="https://github.com/fishaudio/fish-speech.git"
FISH_DIR="/workspace/fish-speech"
# ---

echo "=== [1/9] apt パッケージのインストール ==="
apt update && apt install -y portaudio19-dev libsox-dev ffmpeg

echo "=== [2/9] 基本 Python ライブラリのインストール (Whisper, OpenAI) ==="
# faster-whisper (transcribe_func.py) と openai (answer_generator.py)
pip install librosa soundfile faster-whisper openai

echo "=== [3/9] Hugging Face Hub CLI のインストール ==="
pip install huggingface_hub[cli]

echo "=== [4/9] Hugging Face ログイン (対話型) ==="
echo "!!! 注意: Hugging Face の [read] トークンを入力してください !!!"
huggingface-cli login
# (ここで手動でトークンをペーストする必要があります)

echo "=== [5/9] whisper-streaming リポジトリのクローンまたは更新 ==="
if [ -d "$WHISPER_DIR" ]; then
    echo "$WHISPER_DIR は既に存在します。最新版に更新します..."
    cd "$WHISPER_DIR"
    git pull
else
    echo "$WHISPER_DIR リポジトリをクローンします..."
    cd /workspace
    git clone "$WHISPER_REPO_URL" "$WHISPER_DIR"
fi

echo "=== [6/9] fish-speech リポジトリのクローンまたは更新 ==="
if [ -d "$FISH_DIR" ]; then
    echo "$FISH_DIR は既に存在します。最新版に更新します..."
    cd "$FISH_DIR"
    git pull
else
    echo "$FISH_DIR リポジトリをクローンします..."
    cd /workspace
    git clone "$FISH_REPO_URL" "$FISH_DIR"
fi

echo "=== [7/9] fish-speech のセットアップとモデルダウンロード ==="
# 必ず fish-speech ディレクトリ内で実行します
cd "$FISH_DIR"
echo "[INFO] CWD: $(pwd)"
echo "fish-speech の依存関係をインストールします (時間がかかります)..."
# (cu129の部分はRunPodのCUDAバージョンに合わせて変更してください)
pip install -e .[cu129] 

echo "fish-speech の事前学習済みモデルをダウンロードします..."
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

echo "=== [8/9] whisper-streaming の依存関係インストール ==="
# whisper-streaming の requirements.txt をインストール
cd "$WHISPER_DIR"
echo "[INFO] CWD: $(pwd)"
echo "whisper-streaming の依存関係をインストールします..."
pip install -r requirements.txt

# 元のディレクトリに戻る
cd /workspace
echo "=== [9/9] 全てのセットアップが完了しました。 ==="