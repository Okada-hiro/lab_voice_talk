# /workspace/create_voice_prompt.py
import subprocess
import os
import shutil

# --- 設定 ---
FISH_SPEECH_DIR = "fish-speech"
# 参照音声ファイル (fish-speechディレクトリ内に配置してください)
REFERENCE_AUDIO_NAME = "japanese_sound.wav" 
# 生成するプロンプトファイル (この名前をtext_to_speech.pyでも使います)
PROMPT_TOKEN_FILE_NAME = "fake.npy" 
# ---

def create_prompt():
    """
    参照音声から音声合成用のプロンプトトークン (npy) を生成する
    """
    original_dir = os.getcwd()
    
    if not os.path.isdir(FISH_SPEECH_DIR):
        print(f"[ERROR] {FISH_SPEECH_DIR} が見つかりません。setup_fish_speech.sh を実行しましたか？")
        return
    
    ref_audio_path = os.path.join(FISH_SPEECH_DIR, REFERENCE_AUDIO_NAME)
    if not os.path.exists(ref_audio_path):
        print(f"[ERROR] 参照音声ファイルが見つかりません: {ref_audio_path}")
        print(f"       '{REFERENCE_AUDIO_NAME}' を {FISH_SPEECH_DIR} に配置してください。")
        return

    # コマンド (リポジトリのルートで実行)
    cmd = [
        "python",
        "fish_speech/models/dac/inference.py",
        "-i", REFERENCE_AUDIO_NAME, # ディレクトリ内で実行するため相対パス
        "--checkpoint-path", "checkpoints/openaudio-s1-mini/codec.pth"
    ]
    
    try:
        os.chdir(FISH_SPEECH_DIR)
        print(f"[INFO] CWDを {FISH_SPEECH_DIR} に変更しました。")
        print(f"[INFO] 参照音声からプロンプトトークンを生成します...")
        print(f"[DEBUG] 実行コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[INFO] プロンプト生成成功。")
        print(result.stdout)

        # fish-speechは (入力ファイル名).npy を出力する

        target_npy_path = PROMPT_TOKEN_FILE_NAME 

        if os.path.exists(target_npy_path):
            print(f"[INFO] 生成された {target_npy_path} を {target_npy_path} にリネームします。")
            shutil.move(target_npy_path, target_npy_path)
            print(f"[SUCCESS] {target_npy_path} の準備ができました。")
        else:
            print(f"[ERROR] 期待された出力ファイル {target_npy_path} が見つかりません。")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] プロンプト生成に失敗しました:")
        print(e.stderr)
    except Exception as e:
        print(f"[ERROR] 予期せぬエラー: {e}")
    finally:
        os.chdir(original_dir)
        print(f"[INFO] CWDを {original_dir} に戻しました。")

if __name__ == "__main__":
    create_prompt()