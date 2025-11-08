# new_send_run.py
import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
import time
import requests # requests が必要 (pip install requests)

# --- ★★★★★ 必ず設定してください ★★★★★ ---
# move_files.py で使用していたRunPodのURLから /upload を除いたもの
# (例: "https://8vm9dxp402l5oh-5000.proxy.runpod.net")
RUNPOD_BASE_URL = "https://8vm9dxp402l5oh-5000.proxy.runpod.net" 
# ------------------------------------------

# 録音時間（秒）
RECORD_DURATION = 5
# 回答を待つ最大時間（秒）
POLL_TIMEOUT = 300
# 確認間隔（秒）
POLL_INTERVAL = 2

# ---------------------------
# ① 音声を録音する
# ---------------------------
def record_audio(filename="record.wav", duration=5, fs=44100):
    print(f"{duration}秒間の録音を開始します...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"録音完了: {filename}")
    return filename

# ---------------------------
# ② 録音ファイルをRunPodに送信する
# ---------------------------
def upload_audio(file_path: str, base_url: str) -> bool:
    """
    指定したファイルをRunPodのPodに送信する
    """
    upload_url = f"{base_url}/upload"
    print(f"ファイルをアップロード中: {file_path} -> {upload_url}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(upload_url, files=files, timeout=30)
         
        print(f"サーバ応答: {response.status_code} {response.text}")
        
        if 200 <= response.status_code < 300:
            print("アップロード成功。")
            return True
        else:
            print("アップロード失敗。")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"送信中にネットワークエラーが発生しました: {e}")
        return False
    except Exception as e:
        print(f"送信中に予期せぬエラーが発生しました: {e}")
        return False

# ---------------------------
# ③ 回答音声をダウンロードして再生する
# ---------------------------
def poll_download_and_play(original_filename: str, base_url: str):
    """
    RunPod側で回答ファイルが生成されるのを待ち（ポーリング）、
    ダウンロードして再生する
    """
    
    # watch_and_transcribe.py が生成するファイル名を指定
    answer_filename = original_filename + ".ans.wav"
    download_url = f"{base_url}/download/{answer_filename}"
    local_save_path = f"downloaded_answer_{int(time.time())}.wav"
    
    print(f"回答ファイルの生成を待機中: {answer_filename}")
    
    start_time = time.time()
    file_ready = False
    
    while time.time() - start_time < POLL_TIMEOUT:
        try:
            # ファイルが存在するかどうかをGETリクエストで確認
            # stream=True でヘッダーだけ取得し、すぐに閉じる
            with requests.get(download_url, stream=True, timeout=5) as r:
                if r.status_code == 200:
                    print("回答ファイルを発見！ ダウンロードします。")
                    
                    # --- ダウンロード実行 ---
                    with open(local_save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_ready = True
                    break # whileループを抜ける
                
                else:
                    # 404 Not Found など
                    print(f"  ...まだ準備できていません ({r.status_code})")
            
        except requests.exceptions.RequestException as e:
            print(f"  ...接続エラー ({e})")
        
        time.sleep(POLL_INTERVAL)

    # --- タイムアウトした場合 ---
    if not file_ready:
        print(f"タイムアウト({POLL_TIMEOUT}秒)しました。回答ファイルが見つかりません。")
        return

    # --- 再生 ---
    if file_ready and os.path.exists(local_save_path):
        print(f"回答を再生します: {local_save_path}")
        try:
            # macOS標準の音声再生コマンド 'afplay' を使用
            subprocess.run(["afplay", local_save_path], check=True)
            
            # --- クリーンアップ ---
            print("再生完了。ローカルファイルを削除します。")
            os.remove(local_save_path)
            
        except subprocess.CalledProcessError:
            print(f"エラー: 'afplay' での再生に失敗しました。")
        except FileNotFoundError:
            print("エラー: 'afplay' コマンドが見つかりません。 (macOSで実行していますか？)")
        except Exception as e:
            print(f"再生中にエラー: {e}")

# ---------------------------
# メイン処理
# ---------------------------
if __name__ == "__main__":
    if RUNPOD_BASE_URL == "https://YOUR-POD-ID.proxy.runpod.net":
        print("エラー: スクリプト上部の `RUNPOD_BASE_URL` を、")
        print("       あなたのRunPodのURL（/upload を除く）に設定してください。")
        print(f"       (例: {RUNPOD_BASE_URL} )")
    else:
        # 1. ローカルで録音
        # タイムスタンプでユニークなファイル名にする
        local_wav_file = f"record_{int(time.time())}.wav"
        record_audio(local_wav_file, duration=RECORD_DURATION)
        
        # 2. RunPodにアップロード
        if upload_audio(local_wav_file, RUNPOD_BASE_URL):
            
            # 3. 回答を待ってダウンロード＆再生
            poll_download_and_play(os.path.basename(local_wav_file), RUNPOD_BASE_URL)
        
        # 4. 元の録音ファイルを削除
        if os.path.exists(local_wav_file):
            os.remove(local_wav_file)
            print(f"元の録音ファイルを削除しました: {local_wav_file}")