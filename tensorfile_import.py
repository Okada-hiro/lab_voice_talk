from huggingface_hub import hf_hub_download

# モデルリポジトリID（例: "rinna/japanese-gpt2-medium"）
repo_id = "okadahiroaki/new_model"

# 取得したいファイル名（例: config.json）
filename = "Ref_voice_e300_s2100.safetensors"

# ファイルをローカルにダウンロード
local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="Style_Bert_VITS2/model_assets")

print("保存先:", local_path)