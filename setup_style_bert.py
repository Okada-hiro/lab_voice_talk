# /workspace/setup_stylebert.py
# Style-Bert-TTS のモデルをダウンロードするため、1回だけ実行するスクリプト

from huggingface_hub import hf_hub_download
import os

MODEL_DIR = "model_assets"

#
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

print(f"モデルを '{MODEL_DIR}' にダウンロードします...")
os.makedirs(MODEL_DIR, exist_ok=True)

for file in [model_file, config_file, style_file]:
    print(f"Downloading: {file}")
    try:
        hf_hub_download(
            repo_id="litagin/style_bert_vits2_jvnv", 
            filename=file, 
            local_dir=MODEL_DIR
        )
    except Exception as e:
        print(f"ダウンロード中にエラーが発生しました: {e}")

print("ダウンロード完了。")