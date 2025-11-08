# numpy を pyopenjtalk の想定に合わせる
#pip uninstall -y numpy
#pip install numpy==1.26.4
#pip install style-bert-vits2

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages


bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

from pathlib import Path
from huggingface_hub import hf_hub_download


model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print(file)
    hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="model_assets")


from style_bert_vits2.tts_model import TTSModel

assets_root = Path("model_assets")

model = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cpu",
)

sr, audio = model.infer(
    text="こんにちは。明日の天気に関する情報を提示することはできません。もし他にお手伝いできることがありましたら、ぜひご連絡ください。",
    speaker_id=0,
    style="Neutral",      # 落ち着いたスタイル
    style_weight=0.7,
    pitch_scale=0.75,     # 声を低めに
    intonation_scale=0.3,# 抑揚を少し控えめ
    noise = 0.1,   # デフォルトは0.667くらい
    noise_w = 0.1, # デフォルトは0.8くらい

)
