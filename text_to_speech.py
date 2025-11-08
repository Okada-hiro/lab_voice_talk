# /workspace/text_to_speech.py (v4 - dac/load_model 修正版)
import os
import shutil
import subprocess 
import torch
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import logging

# --- 1. DACモデルのインポートと設定 (inference-2.py に基づく) ---
# (L. 34)
from fish_speech.models.dac.inference import load_model as load_dac_model

# (L. 74, 76)
DAC_CONFIG_NAME = "modded_dac_vq" 
DAC_CHECKPOINT_PATH = "checkpoints/openaudio-s1-mini/codec.pth" 
FISH_SPEECH_DIR = "fish-speech"

# --- 2. Text2Semanticモデルのインポートと設定 (inference.py に基づく) ---
# (L. 367)
from fish_speech.models.text2semantic.inference import (
    init_model as load_t2s_model,
    generate,
    ContentSequence,
    TextPart,
    VQPart,
)

# (L. 612)
T2S_CHECKPOINT_PATH = "checkpoints/openaudio-s1-mini" # T2Sはディレクトリを指定
PROMPT_TOKENS_PATH = "fake.npy" # os.chdir後の相対パス

# --- 共通設定 ---
logger = logging.getLogger(__name__)
DAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ---
# グローバル変数の準備
# ---
GLOBAL_DAC_MODEL = None
GLOBAL_T2S_MODEL = None
GLOBAL_DECODE_FN = None
GLOBAL_PROMPT_TOKENS = None

original_dir_for_load = os.getcwd() # /workspace

try:
    if not os.path.isdir(FISH_SPEECH_DIR):
        raise FileNotFoundError(f"{FISH_SPEECH_DIR} が見つかりません。")
    
    # CWDを /workspace/fish-speech に変更 (全てのパスを解決するため)
    os.chdir(FISH_SPEECH_DIR)
    print(f"[DEBUG] CWDを {FISH_SPEECH_DIR} に変更 (モデルロードのため)")

    # --- 1. DACモデルをグローバルに1回だけロード ---
    # (L. 80)
    print(f"[INFO] 高速化のため、DACモデルをロード中... (Config: {DAC_CONFIG_NAME})")
    GLOBAL_DAC_MODEL = load_dac_model(
        config_name=DAC_CONFIG_NAME, 
        checkpoint_path=DAC_CHECKPOINT_PATH, 
        device=DAC_DEVICE
    )
    print("[INFO] DACモデルロード完了。")

    # --- 2. Text2Semanticモデルをグローバルに1回だけロード ---
    # (L. 620)
    print("[INFO] 高速化のため、Text2Semanticモデルをロード中...")
    GLOBAL_T2S_MODEL, GLOBAL_DECODE_FN = load_t2s_model(
        T2S_CHECKPOINT_PATH, DAC_DEVICE, PRECISION, compile=False
    )
    # (L. 396)
    with torch.device(DAC_DEVICE):
        GLOBAL_T2S_MODEL.setup_caches(
            max_batch_size=1,
            max_seq_len=GLOBAL_T2S_MODEL.config.max_seq_len,
            dtype=PRECISION,
        )
    print("[INFO] Text2Semanticモデルロード完了。")

    # --- 3. 音声プロンプト (fake.npy) をロード ---
    if not os.path.exists(PROMPT_TOKENS_PATH):
        raise FileNotFoundError(f"プロンプトファイル {PROMPT_TOKENS_PATH} が見つかりません。")
    GLOBAL_PROMPT_TOKENS = np.load(PROMPT_TOKENS_PATH)
    print(f"[INFO] 音声プロンプト {PROMPT_TOKENS_PATH} ロード完了。")


except Exception as e:
    print(f"[ERROR] モデルのグローバルロードに失敗しました: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.chdir(original_dir_for_load) # 元のCWD (/workspace) に戻す
    print(f"[DEBUG] CWDを {original_dir_for_load} に戻しました")


# ---
# watch_and_transcribe.py から呼ばれるメイン関数
# ---
@torch.no_grad() # 推論全体で勾配計算を無効化
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = "はっきりと丁寧な音声で読み上げてください。"):
    """
    Fish-Speechでテキストをwavに変換する (v4: 全てのモデルをロード済み)
    """
    
    if GLOBAL_DAC_MODEL is None or GLOBAL_T2S_MODEL is None or GLOBAL_PROMPT_TOKENS is None:
        print("[ERROR] モデルがロードされていません。処理を中断します。")
        return False

    original_dir = os.getcwd() # /workspace
    try:
        os.chdir(FISH_SPEECH_DIR) # CWDを /workspace/fish-speech に変更
        print(f"[DEBUG] CWDを {FISH_SPEECH_DIR} に変更 (推論実行のため)")
        
        # --- 1. Text2Semantic推論 (Python) ---
        # (v3 と同じロジック, inference.py (L. 414-453) に基づく)
        print(f"[DEBUG] 1/2: Text2Semantic推論 (Python)...")
        
        tokenizer = GLOBAL_T2S_MODEL.tokenizer
        base_content_sequence = ContentSequence(modality="interleave")
        
        # 1-1. プロンプト（テキスト＋音声）を追加
        base_content_sequence.append(
            [ TextPart(text=prompt_text), VQPart(codes=torch.from_numpy(GLOBAL_PROMPT_TOKENS)), ],
            add_end=True, speaker=0,
        )
        
        # 1-2. 合成したいテキストを追加
        base_content_sequence.append( [ TextPart(text=text_to_speak), ], add_end=False, speaker=0,)

        # 1-3. モデルへの入力（Tensor）をエンコード
        encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
            tokenizer, num_codebooks=GLOBAL_T2S_MODEL.config.num_codebooks
        )
        encoded = encoded.to(device=DAC_DEVICE)
        prompt_length = encoded.size(1)

        # 1-4. T2Sモデル (generate) を実行
        # (L. 473)
        y = generate(
            model=GLOBAL_T2S_MODEL, prompt=encoded, max_new_tokens=0,
            audio_masks=audio_masks, audio_parts=audio_parts,
            decode_one_token=GLOBAL_DECODE_FN,
            temperature=0.7, top_p=0.7, repetition_penalty=1.5,
        )
        
        # 1-5. セマンティックトークンを抽出
        # (L. 505)
        codes_tensor = y[1:, prompt_length:-1].clone() 
        # (N, D) -> (B=1, D, N) ... dac/inference.py (L. 109) の .ndim == 2 の想定
        semantic_tokens_torch = codes_tensor.unsqueeze(0).to(DAC_DEVICE)
        
        print("[DEBUG] Text2Semantic推論 完了。")

        # --- 2. DAC推論 (Python) ---
        # (inference-2.py (L. 115) のロジック)
        print(f"[DEBUG] 2/2: DACモデルでデコード中 (Python)...")
        
        indices_lens = torch.tensor([semantic_tokens_torch.shape[2]], device=DAC_DEVICE, dtype=torch.long)
        
        # (L. 118)
        wav_tensor, _ = GLOBAL_DAC_MODEL.decode(semantic_tokens_torch, indices_lens)
        wav_np = wav_tensor[0, 0].float().cpu().numpy()
        
        # --- 3. 保存 ---
        # (L. 128)
        sample_rate = GLOBAL_DAC_MODEL.sample_rate
        
        write(output_wav_path, sample_rate, wav_np) # (L. 129) sf.write の代替
        
        print(f"[SUCCESS] 音声を保存しました: {output_wav_path}")

        return True

    except Exception as e:
        print(f"[ERROR] 音声生成中に例外: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラーを表示
        return False

    finally:
        os.chdir(original_dir) # 必ず /workspace に戻す
        print(f"[DEBUG] CWDを {original_dir} に戻しました")

# (単体テスト部分は v3 と同じなので省略)


# --- 単体テスト ---
if __name__ == "__main__":
    print("--- 完全高速音声合成 単体テスト ---")
    if GLOBAL_DAC_MODEL is not None and GLOBAL_T2S_MODEL is not None:
        TEST_TEXT = "こんにちは。これはfish-speechの完全高速化テストです。"
        TEST_OUTPUT = "/workspace/test_output_v3.wav"
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # 1回目（ウォームアップ）
        print("\n--- 1回目 (ウォームアップ) ---")
        synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        
        # 2回目（速度計測）
        print("\n--- 2回目 (速度計測) ---")
        start_time.record()
        success = synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        end_time.record()
        
        torch.cuda.synchronize() # GPU処理の完了を待つ
        
        if success:
            print(f"[SUCCESS] テストファイルが生成されました: {TEST_OUTPUT}")
            print(f"***** 2回目の実行時間: {start_time.elapsed_time(end_time) / 1000.0:.3f} 秒 *****")
        else:
            print("[FAIL] テストに失敗しました。")
    else:
        print("[SKIP] モデルがロードされていないため、テストをスキップします。")