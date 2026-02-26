# /workspace/transcribe_func.py (v3 - whisper_online.py ベース)
import numpy as np
import librosa
from functools import lru_cache
import logging
import sys
from faster_whisper import WhisperModel # faster-whisper はインストール済みと仮定

# --- whisper_online.py から必要な部分を抜粋 ---

logger = logging.getLogger(__name__)

# (L.18)
@lru_cache(10**6)
def load_audio(fname):
    """
    librosa を使ってファイルをロードし、16kHzにリサンプリングする
    """
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

# (L.30)
class ASRBase:
    sep = " " 
    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto": self.original_language = None
        else: self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)
    def load_model(self, modelsize, cache_dir, model_dir): raise NotImplementedError
    def transcribe(self, audio, init_prompt=""): raise NotImplementedError
    def use_vad(self): raise NotImplementedError

# (L.107)
class FasterWhisperASR(ASRBase):
    """
    whisper_online.py から抜粋した FasterWhisperASR クラス。
    .transcribe メソッド（オブジェクトのメソッド）を持つ。
    """
    sep = "" # (L.110)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        if model_dir is not None:
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")
        
        # (L.125)
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)
        return model

    def transcribe(self, audio, init_prompt=""):
        # (L.141)
        segments, info = self.model.transcribe(
            audio, 
            language=self.original_language, 
            initial_prompt=init_prompt, 
            beam_size=5, 
            word_timestamps=True, 
            condition_on_previous_text=True, 
            **self.transcribe_kargs
        )
        return list(segments)

    def ts_words(self, segments):
        # (L.153)
        o = []
        for segment in segments:
            # (L.156)
            if segment.no_speech_prob > 0.9: 
                logger.debug(f"無音セグメントをスキップ (Prob: {segment.no_speech_prob:.2f})")
                continue
            for word in segment.words:
                o.append((word.start, word.end, word.word))
        return o

# --- 抜粋ここまで ---


# --- ★ 1. グローバル・モデル・ローディング ---
# (watch_and_transcribe.py が import した時に1回だけ実行される)
print("[INFO] Whisper (v3) グローバルロード: FasterWhisperASRモデル 'medium' をロード中...")
GLOBAL_ASR_MODEL_INSTANCE = None
try:
    # watch_and_transcribe.py の設定に合わせる
    GLOBAL_ASR_MODEL_INSTANCE = FasterWhisperASR(lan="ja", modelsize="medium")
    print("[INFO] Whisper (v3) グローバルロード: モデル（オブジェクト）ロード完了。")
except Exception as e:
    print(f"[ERROR] Whisper (v3) グローバルロード: モデルロードに失敗: {e}")
    import traceback
    traceback.print_exc()

# --- ★ 2. watch_and_transcribe.py が呼び出すためのラッパー関数 ---

def whisper_text_only(audio_path: str, language: str = "ja", output_txt: str = None) -> str:
    """
    プリロードされた GLOBAL_ASR_MODEL_INSTANCE を使って文字起こしを実行する
    """
    
    if GLOBAL_ASR_MODEL_INSTANCE is None:
        print("[ERROR] Whisperモデルがロードされていません。処理を中断します。")
        return ""
    
    # (L.18)
    try:
        # 1. librosa で 16kHz にリサンプリング
        audio_data = load_audio(audio_path)
    except Exception as e:
        print(f"[ERROR] librosa での音声ファイル読み込みに失敗: {e}")
        return ""

    try:
        # 2. transcribe (オブジェクトのメソッドを呼び出す)
        # (L.141)
        segments = GLOBAL_ASR_MODEL_INSTANCE.transcribe(audio_data, init_prompt="")
        
        # 3. ts_words (オブジェクトのメソッドを呼び出す)
        # (L.153)
        word_list_tuples = GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)
        
        # 4. FasterWhisperASR.sep = "" (L.110) に基づき結合
        all_text = GLOBAL_ASR_MODEL_INSTANCE.sep.join([word[2] for word in word_list_tuples]).strip()

    except Exception as e:
        print(f"[ERROR] faster-whisper推論中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return ""

    # 5. テキストファイルへの保存
    if output_txt:
        try:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(all_text)
        except Exception as e:
            print(f"[ERROR] 文字起こしテキストの保存に失敗: {e}")
            
    return all_text