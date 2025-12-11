# /workspace/new_text_to_speech.py (v3 - Wrapper統合版)
import torch
from pathlib import Path
from scipy.io.wavfile import write
import scipy.signal
import os
import numpy as np
import sys
import json
import io

# ★追加: アクセント解析用
import pyopenjtalk

# --- 1. Style-Bert-VITS2 リポジトリのルートを sys.path に追加 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2") 

if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"[INFO] Added to sys.path: {REPO_PATH}")

# --- 2. Style-Bert-TTS のインポート ---
try:
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
except ImportError as e:
    print(f"[ERROR] Style-Bert-TTS のインポートに失敗しました。")
    print(f"       REPO_PATH ({REPO_PATH}) が 'Style_Bert_VITS2' として存在するか確認してください。")
    raise

# --- グローバル変数の準備 ---
GLOBAL_TTS_MODEL = None
GLOBAL_SPEAKER_ID = None
GLOBAL_ACCENT_RULES = {} # ★追加: アクセント辞書用
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 設定パラメータ (アナウンサー風プリセット) ---
FT_SPEAKER_NAME = "Ref_voice" 
FT_MODEL_FILE = "Ref_voice_e300_s2100.safetensors"
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
ACCENT_JSON_FILE = "accents.json" # ★追加

# --- ★追加: ヘルパー関数群 (Wrapperから移植) ---

def load_accent_dict(json_path):
    """アクセント辞書をロードしてグローバル変数に格納"""
    global GLOBAL_ACCENT_RULES
    if not os.path.exists(json_path):
        print(f"[WARNING] Accent JSON not found: {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for word, tones in data.items():
            phones = pyopenjtalk.g2p(word, kana=False).split(" ")
            phones = [p for p in phones if p not in ('pau', 'sil')]
            GLOBAL_ACCENT_RULES[word] = {"phones": phones, "tones": tones}
            count += 1
        print(f"[INFO] Loaded {count} accent rules from {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load accent dict: {e}")

def _parse_openjtalk_accent(labels):
    """OpenJTalkのラベルから音素とアクセントを解析"""
    phones = []
    tones = []
    for label in labels:
        parts = label.split('/')
        p3 = label.split('-')[1].split('+')[0]
        if p3 == 'sil': p3 = 'pau'
        phones.append(p3)
        if p3 == 'pau':
            tones.append(0)
            continue
        try:
            a_part = parts[1]
            if 'A:' not in a_part:
                tones.append(0)
                continue
            nums = a_part.split(':')[1].split('+')
            a1 = int(nums[0])
            a2 = int(nums[1])
            is_high = 0
            if a1 == 0:
                if a2 == 1: is_high = 0
                else:       is_high = 1
            else:
                if a2 <= a1:
                    if a2 == 1 and a1 > 1: is_high = 0
                    else: is_high = 1
                else:
                    is_high = 0
            tones.append(is_high)
        except:
            tones.append(0)
    return phones, tones

def _g2p_and_patch(text):
    """テキストを音素に変換し、辞書に基づいてアクセントを修正する"""
    # OpenJTalkで標準の読みとアクセントを取得
    labels = pyopenjtalk.extract_fullcontext(text)
    phones, tones = _parse_openjtalk_accent(labels)

    # 辞書ルールを適用 (Patching)
    for word, rule in GLOBAL_ACCENT_RULES.items():
        target_phones = rule['phones']
        target_tones = rule['tones']
        
        if len(target_phones) != len(target_tones):
            continue

        seq_len = len(target_phones)
        # 音素列の中から単語の音素列を探して、トーンを上書き
        for i in range(len(phones) - seq_len + 1):
            if phones[i : i + seq_len] == target_phones:
                for j, t_val in enumerate(target_tones):
                    tones[i + j] = t_val

    return phones, tones

def _apply_lowpass_scipy(audio_numpy, sr, cutoff):
    """Scipyを使ったローパスフィルタ (金属音除去)"""
    if cutoff <= 0 or cutoff >= sr / 2:
        return audio_numpy

    audio_numpy = np.squeeze(audio_numpy)
    try:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        sos = scipy.signal.butter(5, normal_cutoff, btype='low', analog=False, output='sos')
        filtered = scipy.signal.sosfilt(sos, audio_numpy)
        return filtered
    except Exception as e:
        print(f"[ERROR] Scipy filter failed: {e}")
        return audio_numpy

# --- 初期化処理 ---
try:
    print(f"[INFO] Style-Bert-TTS (FT): Loading models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    assets_root = Path(REPO_PATH) / "model_assets"
    model_path = assets_root / FT_MODEL_FILE
    config_path = assets_root / FT_CONFIG_FILE
    style_vec_path = assets_root / FT_STYLE_FILE

    if not all([model_path.exists(), config_path.exists(), style_vec_path.exists()]):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    GLOBAL_TTS_MODEL = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    
    # 話者ID取得
    GLOBAL_SPEAKER_ID = GLOBAL_TTS_MODEL.spk2id[FT_SPEAKER_NAME]
    
    # ★追加: アクセント辞書のロード
    load_accent_dict(os.path.join(WORKSPACE_DIR, ACCENT_JSON_FILE))

    # ウォームアップ
    print("[INFO] Warm-up inference...")
    _ = GLOBAL_TTS_MODEL.infer(
        text="あ",
        language=Languages.JP,
        speaker_id=GLOBAL_SPEAKER_ID,
        length=0.1
    )
    print("[INFO] Initialization complete.")

except Exception as e:
    print(f"[ERROR] Init failed: {e}")
    import traceback
    traceback.print_exc()

# --- パラメータ設定 (ここを調整するとチャットボットの声が変わります) ---
DEFAULT_PARAMS = {
    "style": "Neutral",
    "style_weight": 0.1,       # 0.1 (辞書優先、スタイルは弱め)
    "pitch": 1.2,              # 1.2 (少し高く、明るく)
    "intonation": 1.3,         # 1.3 (抑揚をつける)
    "length": 0.9,             # 0.9 (少し早口に、テキパキと)
    "sdp_ratio": 0.0,          # 0.0 (ランダム性を排除し安定させる)
    "noise": 0.6,
    "noise_w": 0.8,
    "assist_text": "アナウンサーです。はきはきと、明瞭に喋ります。全く雑音のない、クリアな音声で喋ります。。",
    "assist_text_weight": 0.2,
    "lpf_cutoff": 9000         # 9000Hz以上をカット (ノイズ除去)
}

# --- 音声合成関数 (ファイル保存版) ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    if GLOBAL_TTS_MODEL is None:
        return False
        
    try:
        # ★処理1: アクセント修正 (G2P Patching)
        phones, tones = _g2p_and_patch(text_to_speak)
        
        # ★処理2: 推論 (パラメータ適用)
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            given_phone=phones, # 修正済み音素を使用
            given_tone=tones,   # 修正済みアクセントを使用
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style=DEFAULT_PARAMS["style"],
            style_weight=DEFAULT_PARAMS["style_weight"],
            pitch_scale=DEFAULT_PARAMS["pitch"],
            intonation_scale=DEFAULT_PARAMS["intonation"],
            length=DEFAULT_PARAMS["length"],
            sdp_ratio=DEFAULT_PARAMS["sdp_ratio"],
            noise=DEFAULT_PARAMS["noise"],
            noise_w=DEFAULT_PARAMS["noise_w"],
            assist_text=DEFAULT_PARAMS["assist_text"],
            assist_text_weight=DEFAULT_PARAMS["assist_text_weight"],
            use_assist_text=True
        )
        
        # 正規化 (float scaleに戻す)
        if not isinstance(audio_data, np.ndarray):
            audio_data = audio_data.cpu().numpy()
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.5:
            audio_data = audio_data / 32768.0

        # ★処理3: ローパスフィルタ (ノイズ除去)
        audio_data = _apply_lowpass_scipy(audio_data, sr, DEFAULT_PARAMS["lpf_cutoff"])

        # int16変換
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        write(output_wav_path, sr, audio_int16)
        print(f"[SUCCESS] Saved to {output_wav_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 音声合成関数 (メモリ返却版: Chatbot用) ---
def synthesize_speech_to_memory(text_to_speak: str) -> bytes:
    if GLOBAL_TTS_MODEL is None:
        return None
        
    try:
        # ★処理1: アクセント修正
        phones, tones = _g2p_and_patch(text_to_speak)

        # ★処理2: 推論
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            given_phone=phones,
            given_tone=tones,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style=DEFAULT_PARAMS["style"],
            style_weight=DEFAULT_PARAMS["style_weight"],
            pitch_scale=DEFAULT_PARAMS["pitch"],
            intonation_scale=DEFAULT_PARAMS["intonation"],
            length=DEFAULT_PARAMS["length"],
            sdp_ratio=DEFAULT_PARAMS["sdp_ratio"],
            noise=DEFAULT_PARAMS["noise"],
            noise_w=DEFAULT_PARAMS["noise_w"],
            assist_text=DEFAULT_PARAMS["assist_text"],
            assist_text_weight=DEFAULT_PARAMS["assist_text_weight"],
            use_assist_text=True
        )

        # 正規化
        if not isinstance(audio_data, np.ndarray):
            audio_data = audio_data.cpu().numpy()
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.5:
            audio_data = audio_data / 32768.0

        # ★処理3: ローパスフィルタ (リサンプリング前に適用するのが良い)
        audio_data = _apply_lowpass_scipy(audio_data, sr, DEFAULT_PARAMS["lpf_cutoff"])

        # リサンプリング (Chatbot用に16kHzへ)
        target_sr = 16000
        if sr > target_sr:
            num_samples = int(len(audio_data) * float(target_sr) / sr)
            audio_resampled = scipy.signal.resample(audio_data, num_samples)
            audio_data = audio_resampled

        # int16変換
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        return audio_int16.tobytes()

    except Exception as e:
        print(f"[ERROR] Memory Synthesis Error: {e}")
        return None

# --- 単体テスト ---
if __name__ == "__main__":
    print("\n--- Style-Bert-TTS (Integrated Version) Test ---")
    if GLOBAL_TTS_MODEL:
        # テスト: 辞書にある単語を含めると効果がわかります
        TEST_TEXT = "こんにちは。駅前のカフェに行きましょう。パソコン作業も捗りますよ。"
        TEST_OUTPUT = "test_integrated_output.wav"
        
        synthesize_speech(TEST_TEXT, TEST_OUTPUT)