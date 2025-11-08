# pod_server.py (修正後)
from flask import Flask, request
import os # os をインポート

app = Flask(__name__)

# 保存先のディレクトリを /incoming_audio に固定
SAVE_DIR = "incoming_audio"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    # 保存ファイル名を received_FILENAME にする
    filename = f"received_{file.filename}"
    save_path = os.path.join(SAVE_DIR, filename)
    
    try:
        file.save(save_path)
        print(f"File saved to: {save_path}") # ログ出力
        return f"File received and saved to {save_path}!", 200
    except Exception as e:
        print(f"File save error: {e}")
        return f"Error saving file: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)