# /workspace/new_speaker_filter.py (メモリ処理対応版)
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os
import logging

logger = logging.getLogger(__name__)

# --- 1. 音声読み込み関数 (既存互換) ---
def load_audio(path: str, target_sample_rate=16000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    signal, fs = torchaudio.load(path)
    # ステレオ→モノラル
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    # リサンプリング
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)
    return signal

# --- 2. 声紋フィルタークラス ---
class SpeakerGuard:
    def __init__(self):
        print("⏳ [SpeakerGuard] モデルをロード中... (SpeechBrain)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.allowed_embeddings = [] 
        self.threshold = 0.35 
        print(f"✅ [SpeakerGuard] 準備完了 (Device: {self.device})")

    def extract_embedding(self, audio_tensor):
        # 入力テンソルをモデルと同じデバイスへ
        audio_tensor = audio_tensor.to(self.device)
        
        # バッチ次元がない場合 (samples,) -> (1, samples)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # 長さ情報の作成 (今回はバッチ1なので全長1.0)
        wav_lens = torch.ones(audio_tensor.shape[0]).to(self.device)

        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_tensor, wav_lens)
        return embedding

    def _check_similarity(self, current_embedding) -> bool:
        """共通の判定ロジック"""
        # 初回登録
        if not self.allowed_embeddings:
            print("🔒 [SpeakerGuard] 最初の話者をオーナーとして自動登録しました")
            self.allowed_embeddings.append(current_embedding)
            return True

        max_score = -1.0
        is_match = False

        for saved_emb in self.allowed_embeddings:
            score = torch.nn.functional.cosine_similarity(
                saved_emb, current_embedding, dim=-1
            )
            score_val = score.item()
            if score_val > max_score:
                max_score = score_val
            
            if score_val > self.threshold:
                is_match = True
                break 

        if is_match:
            logger.info(f"✅ [SpeakerGuard] 本人確認OK (スコア: {max_score:.4f})")
        else:
            logger.info(f"🚫 [SpeakerGuard] ブロック (最大スコア: {max_score:.4f})")
            
        return is_match

    def register_new_speaker(self, audio_path: str) -> bool:
        try:
            audio_tensor = load_audio(audio_path)
            new_emb = self.extract_embedding(audio_tensor)
            self.allowed_embeddings.append(new_emb)
            print(f"📝 [SpeakerGuard] 新しい話者を登録しました (現在 {len(self.allowed_embeddings)} 人)")
            return True
        except Exception as e:
            print(f"[SpeakerGuard Error] 登録失敗: {e}")
            return False

    def is_owner(self, audio_path: str) -> bool:
        """(旧) ファイルパスから判定"""
        try:
            audio_tensor = load_audio(audio_path)
            current_embedding = self.extract_embedding(audio_tensor)
            return self._check_similarity(current_embedding)
        except Exception as e:
            print(f"[SpeakerGuard Error] 読み込み失敗: {e}")
            return False

    def verify_tensor(self, audio_tensor: torch.Tensor) -> bool:
        """(新) メモリ上のTensorから判定 (高速)"""
        try:
            current_embedding = self.extract_embedding(audio_tensor)
            return self._check_similarity(current_embedding)
        except Exception as e:
            print(f"[SpeakerGuard Error] Tensor判定失敗: {e}")
            return False