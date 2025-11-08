# /workspace/text_to_speech.py (v-StyleBert, 正規化対応版)
# Style-Bert-TTS をグローバルロードするバージョン

import torch
from pathlib import Path
from scipy.io.wavfile import write # .wav ファイル保存
import os
import numpy as np # ★ 1. NumPy をインポート

# --- Style-Bert-TTS のインポート ---
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

# ---
# グローバル変数の準備 (変更なし)
# ---
GLOBAL_TTS_MODEL = None
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # --- 1. BERTモデルのグローバルロード (変更なし) ---
    print("[INFO] Style-Bert-TTS: Loading BERT models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    print("[INFO] Style-Bert-TTS: BERT models loaded.")

    # --- 2. TTSモデルのグローバルロード (変更なし) ---
    print("[INFO] Style-Bert-TTS: Loading TTSModel...")
    assets_root = Path("model_assets")
    
    model_file = assets_root / "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    config_file = assets_root / "jvnv-F1-jp/config.json"
    style_file = assets_root / "jvnv-F1-jp/style_vectors.npy"

    if not all([model_file.exists(), config_file.exists(), style_file.exists()]):
        raise FileNotFoundError(f"モデルファイルが見つかりません。先に setup_style_bert.py を実行しましたか？")

    GLOBAL_TTS_MODEL = TTSModel(
        model_path=model_file,
        config_path=config_file,
        style_vec_path=style_file,
        device=GLOBAL_DEVICE
    )
    print("[INFO] Style-Bert-TTS: TTSModel loaded. All models ready.")

except Exception as e:
    print(f"[ERROR] Style-Bert-TTS モデルのグローバルロードに失敗しました: {e}")
    import traceback
    traceback.print_exc()

# ---
# watch_and_transcribe.py から呼ばれるメイン関数
# ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    """
    Style-Bert-TTS でテキストをwavに変換する
    """
    if GLOBAL_TTS_MODEL is None:
        print("[ERROR] Style-Bert-TTS モデルがロードされていません。")
        return False
        
    try:
        print(f"[DEBUG] Style-Bert-TTS: 音声合成開始... '{text_to_speak[:20]}...'")
        
        # 1. 推論の実行
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            speaker_id=0,
            style="Neutral",
            style_weight=0.7,
            pitch_scale=0.75,
            intonation_scale=0.3,
            noise = 0.1,
            noise_w = 0.1,
        )
        
        # --- ★ 2. 16-bit PCM への変換 (ご指摘のロジック) ---
        if audio_data.dtype != np.int16:
            # 正規化して16bitに変換
            audio_norm = audio_data / np.abs(audio_data).max()  # -1.0 ~ 1.0
            audio_int16 = (audio_norm * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
        # --- ★ 修正ここまで ---

        # 3. WAV ファイルとして保存 (変換後のデータを書き込む)
        write(output_wav_path, sr, audio_int16)
        
        print(f"[SUCCESS] Style-Bert-TTS 音声を保存しました: {output_wav_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Style-Bert-TTS 音声生成中に例外: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 単体テスト (変更なし) ---
if __name__ == "__main__":
    print("\n--- Style-Bert-TTS 単体テスト ---")
    
    if GLOBAL_TTS_MODEL is None:
        print("[FAIL] モデルのグローバルロードに失敗したため、テストを中止します。")
    else:
        TEST_TEXT = "こんにちは。これは、Style-Bert-TTS の単体テストです。"
        TEST_OUTPUT = "/workspace/test_stylebert_output.wav"
        
        print(f"テキスト: {TEST_TEXT}")
        print(f"出力先: {TEST_OUTPUT}")
        
        if os.path.exists(TEST_OUTPUT):
            os.remove(TEST_OUTPUT)
            
        success = synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        
        if success and os.path.exists(TEST_OUTPUT):
            print(f"[SUCCESS] テストファイルが正常に生成されました。")
        else:
            print("[FAIL] テストに失敗しました。")