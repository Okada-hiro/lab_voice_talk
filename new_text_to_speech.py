# /workspace/text_to_speech.py (v-FineTuned, v2 - パスフラット化)
import torch
from pathlib import Path
from scipy.io.wavfile import write
import scipy.signal
import os
import numpy as np
import sys

# --- ★ここから修正 (ここから) ---

# --- 1. Style-Bert-VITS2 リポジトリのルートを sys.path に追加 ---
# このファイル (new_text_to_speech.py) があるディレクトリ (= /workspace)
WORKSPACE_DIR = os.getcwd()
# git clone したリポジトリのパス
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2") 

# sys.path にリポジトリのルートを追加
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"[INFO] Added to sys.path: {REPO_PATH}")
# ---

# --- 2. Style-Bert-TTS のインポート (パス修正) ---
try:
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.nlp import bert_models
    # 変更後:
    from style_bert_vits2.nlp import bert_models
    
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.constants import Languages
    # 変更後:
    from style_bert_vits2.constants import Languages
    
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.tts_model import TTSModel
    # 変更後:
    from style_bert_vits2.tts_model import TTSModel
    
except ImportError as e:
    print(f"[ERROR] Style-Bert-TTS のインポートに失敗しました。")
    print(f"       REPO_PATH ({REPO_PATH}) が 'Style_Bert_VITS2' として存在するか確認してください。")
    print(f"       エラー詳細: {e}")
    # プログラムを停止させるためにエラーを再送出
    raise

# --- ★ここまで修正 (ここまで) ---

# --- グローバル変数の準備 (変更なし) ---
GLOBAL_TTS_MODEL = None
GLOBAL_SPEAKER_ID = None
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ファインチューニングモデルの設定 ---
# 話者名 (config.json の spk2id 内) は "Ref_voice" のまま
FT_SPEAKER_NAME = "Ref_voice" 
# model_assets/ 直下に置くファイル名
FT_MODEL_FILE = "Ref_voice_e3_s936.safetensors"
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
# ---

try:
    # --- BERTモデルのグローバルロード (変更なし) ---
    print(f"[INFO] Style-Bert-TTS (FT): Loading BERT models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    print("[INFO] Style-Bert-TTS (FT): BERT models loaded.")

    # --- ★ 修正: TTSモデルのロードパス ---
    print("[INFO] Style-Bert-TTS (FT): Loading Fine-Tuned TTSModel...")
    assets_root = Path(REPO_PATH) / "model_assets"
    
    # のご要望通り、サブディレクトリ (Ref_voice/) を使わないパスに変更
    model_path = assets_root / FT_MODEL_FILE
    config_path = assets_root / FT_CONFIG_FILE
    style_vec_path = assets_root / FT_STYLE_FILE

    if not all([model_path.exists(), config_path.exists(), style_vec_path.exists()]):
        print(f"[DEBUG] Check Failed (Path: {assets_root}):")
        print(f"  Model:  {model_path} - Exists: {model_path.exists()}")
        print(f"  Config: {config_path} - Exists: {config_path.exists()}")
        print(f"  Style:  {style_vec_path} - Exists: {style_vec_path.exists()}")
        raise FileNotFoundError(f"モデルファイルが 'model_assets/' 直下に見つかりません。")

    GLOBAL_TTS_MODEL = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    print("[INFO] Style-Bert-TTS (FT): TTSModel loaded.")

    # --- ★ 話者IDの取得 (話者名は 'Ref_voice' で固定) ---
    try:
        GLOBAL_SPEAKER_ID = GLOBAL_TTS_MODEL.spk2id[FT_SPEAKER_NAME]
        print(f"[INFO] Style-Bert-TTS (FT): Found speaker: {FT_SPEAKER_NAME} (ID: {GLOBAL_SPEAKER_ID})")
    except KeyError:
        print(f"[ERROR] 話者 '{FT_SPEAKER_NAME}' が {config_path} (spk2id) に見つかりません。")
        print(f"利用可能な話者: {list(GLOBAL_TTS_MODEL.spk2id.keys())}")
        raise

    # --- ★追加: ウォームアップ処理 (ここから) ---
    print("[INFO] Style-Bert-TTS (FT): Performing Warm-up (dummy inference)...")
    try:
        # 「あ」と一瞬だけ生成させて、CUDAの初期化コストをここで払っておく
        # 結果は使わないので捨てる
        _ = GLOBAL_TTS_MODEL.infer(
            text="あ",
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=0.1 # 最短で終わらせる
        )
        print("[INFO] Style-Bert-TTS (FT): Warm-up complete! (Ready for fast inference)")
    except Exception as wu_e:
        print(f"[WARNING] Warm-up failed (will proceed anyway): {wu_e}")
    # --- ★追加: ウォームアップ処理 (ここまで) ---
    print("[INFO] Style-Bert-TTS (FT): All models ready.")

except Exception as e:
    print(f"[ERROR] Style-Bert-TTS (FT) モデルのグローバルロードに失敗しました: {e}")
    import traceback
    traceback.print_exc()

# --- synthesize_speech 関数 (変更なし) ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    if GLOBAL_TTS_MODEL is None or GLOBAL_SPEAKER_ID is None:
        print("[ERROR] Style-Bert-TTS (FT) モデルがロードされていません。")
        return False
        
    try:
        print(f"[DEBUG] Style-Bert-TTS (FT): 音声合成開始... '{text_to_speak[:20]}...'")
        
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID, # グローバルな話者IDを使用
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0
        )
        
        # 16-bit PCM への変換
        if audio_data.dtype != np.int16:
            audio_norm = audio_data / np.abs(audio_data).max()
            audio_int16 = (audio_norm * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # WAV ファイルとして保存
        write(output_wav_path, sr, audio_int16)
        
        print(f"[SUCCESS] Style-Bert-TTS (FT) 音声を保存しました: {output_wav_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Style-Bert-TTS (FT) 音声生成中に例外: {e}")
        import traceback
        traceback.print_exc()
        return False

import io
from scipy.io.wavfile import write as scipy_write

def synthesize_speech_to_memory(text_to_speak: str) -> bytes:
    """
    音声をファイルに保存せず、バイトデータとして直接返す（Scipy高速化版）
    """
    if GLOBAL_TTS_MODEL is None or GLOBAL_SPEAKER_ID is None:
        return None
        
    try:
        # 1. 推論実行
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0
        )
        
        # 2. 16bit PCMに変換 (正規化)
        if audio_data.dtype != np.int16:
            audio_norm = audio_data / np.abs(audio_data).max()
            # floatのままリサンプリングするためにここではまだint16にしない
            audio_float = audio_norm
        else:
            audio_float = audio_data.astype(np.float32) / 32768.0

        # --- ★高速化ポイント: Scipyでリサンプリング (エラー回避版) ---
        target_sr = 16000
        if sr > target_sr:
            # サンプル数を計算
            num_samples = int(len(audio_float) * float(target_sr) / sr)
            # Scipyでリサンプリング (librosaよりトラブルが少ない)
            audio_resampled = scipy.signal.resample(audio_float, num_samples)
            
            # int16に変換
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            sr = target_sr
        else:
            # リサンプリング不要な場合
            audio_int16 = (audio_float * 32767).astype(np.int16)
        # -------------------------------------------------------
        # 新しいコード（Raw PCM返却）:
        # tobytes() でメモリ上の配列をそのままバイナリ化します
        return audio_int16.tobytes()
        

    except Exception as e:
        print(f"[ERROR] Memory Synthesis Error: {e}")
        # エラー時はNoneを返すか、元のファイルを返すなど安全策をとる
        return None
    
# --- 単体テスト (変更なし) ---
if __name__ == "__main__":
    print("\n--- Style-Bert-TTS (FineTuned, FlatPath) 単体テスト ---")
    
    if GLOBAL_TTS_MODEL is None:
        print("[FAIL] モデルのグローバルロードに失敗したため、テストを中止します。")
    else:
        TEST_TEXT = "こんにちは。これは、ファインチューニングしたモデルによる音声合成のテストです。"
        TEST_OUTPUT = "/workspace/test_finetuned_output.wav"
        
        print(f"テキスト: {TEST_TEXT}")
        print(f"出力先: {TEST_OUTPUT}")
        
        if os.path.exists(TEST_OUTPUT):
            os.remove(TEST_OUTPUT)
            
        success = synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        
        if success and os.path.exists(TEST_OUTPUT):
            print(f"[SUCCESS] テストファイルが正常に生成されました。")
        else:
            print("[FAIL] テストに失敗しました。")