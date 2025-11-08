# /workspace/watch_and_transcribe.py
import time
import os
from transcribe_func import whisper_text_only 
from answer_generator import generate_answer
from text_to_speech_new import synthesize_speech

WATCH_DIR = "incoming_audio"
OUTPUT_DIR = "outgoing_audio" # ★ 追加: 出力用ディレクトリ

# --- モデルのグローバルロード ---
# (transcribe_func.py と text_to_speech.py の import 時に自動でロードされる)
print("[INFO] Whisper/Fish-Speech モデルはグローバルロード済み。")

# ★ 追加: 念のため両方のディレクトリを作成
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] 監視ループを開始します。")
# ---

already_seen = set()

while True:
    files = os.listdir(WATCH_DIR)
    for f in files:
        if f not in already_seen and f.endswith(".wav"):
            audio_path = os.path.join(WATCH_DIR, f)
            print(f"[INFO] 新しい音声を検知: {audio_path}")

            # --- 1. 文字起こし ---
            # (文字起こし結果(txt)は入力ファイルと同じ場所(WATCH_DIR)に保存)
            output_txt_path = os.path.join(WATCH_DIR, f + ".txt")
            print(f"[INFO] 文字起こしを開始... )")
            
            question_text = whisper_text_only(
                audio_path,
                language="ja",
                output_txt=output_txt_path
            )
            
            print(f"[INFO] 文字起こし完了: {output_txt_path}")
            print(f"[INFO] 質問内容: {question_text}")

            # --- 2. 回答生成 (変更なし) ---
            print(f"[INFO] 回答生成を開始...")
            answer_text = generate_answer(question_text)
            print(f"[INFO] 回答: {answer_text}")

            # --- 3. 回答の保存 (★ 修正) ---
            # ★ 変更: 保存先を OUTPUT_DIR に変更
            answer_file_path = os.path.join(OUTPUT_DIR, f + ".ans.txt")
            try:
                with open(answer_file_path, "w", encoding="utf-8") as ans_f:
                    ans_f.write(answer_text)
                print(f"[INFO] 回答を保存: {answer_file_path}") # パスが変更された
            except Exception as e:
                print(f"[ERROR] 回答ファイルの保存に失敗しました: {e}")

            # --- 4. 回答の音声合成 (★ 修正) ---
            print(f"[INFO] 回答の音声合成を開始...")
            # ★ 変更: 保存先を OUTPUT_DIR に変更
            answer_wav_path_abs = os.path.abspath(os.path.join(OUTPUT_DIR, f + ".ans.wav"))
            
            success_tts = synthesize_speech(
                text_to_speak=answer_text,
                output_wav_path=answer_wav_path_abs
            )
            
            if success_tts:
                print(f"[INFO] 音声合成 完了: {answer_wav_path_abs}") # パスが変更された
            else:
                print(f"[WARN] 音声合成に失敗しました。")

            already_seen.add(f)

    time.sleep(1)  # 1秒ごとにチェック