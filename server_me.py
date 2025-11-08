from flask import Flask, request, send_from_directory
import os
import uuid

app = Flask(__name__)

# ★ 変更: ディレクトリを2つに分離
UPLOAD_DIR = "incoming_audio"
DOWNLOAD_DIR = "outgoing_audio"

os.makedirs(UPLOAD_DIR, exist_ok=True)  # アップロード用ディレクトリ
os.makedirs(DOWNLOAD_DIR, exist_ok=True) # ダウンロード用ディレクトリ

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    filename = file.filename
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    
    safe_filename = filename

    # ★ 変更: UPLOAD_DIR に保存
    save_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        file.save(save_path)
        print(f"File saved to: {save_path}")
        return f"File received and saved to {save_path}!", 200
    except Exception as e:
        print(f"File save error: {e}")
        return f"Error saving file: {e}", 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    ★ 変更: outgoing_audio フォルダからファイルをダウンロードさせる
    """
    print(f"Download request received for: {filename}")
    try:
        # ★ 変更: DOWNLOAD_DIR から送信
        return send_from_directory(
            DOWNLOAD_DIR,
            filename,
            as_attachment=True
        )
    except FileNotFoundError:
        print(f"Download request failed: File not found {filename}")
        return "File not found", 404
    except Exception as e:
        print(f"Download error: {e}")
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)