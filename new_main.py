# /workspace/main.py
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import os
import asyncio
import time
import subprocess # ffmpeg 実行のため
import logging # ★ 優先度1: ロギングモジュールをインポート
import sys # ★ 優先度1: ロギング出力先指定のため

# --- 既存の処理モジュールをインポート ---
try:
    from transcribe_func import whisper_text_only
    from new_answer_generator import generate_answer
    from new_text_to_speech import synthesize_speech
except ImportError:
    print("[ERROR] 必要なモジュール(transcribe_func, answer_generator, new_text_to_speech)が見つかりません。")
    exit(1)

# --- ★ 優先度1: ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # コンソールに標準出力する
)
logger = logging.getLogger(__name__)

# --- 設定 ---
PROCESSING_DIR = "incoming_audio" # アップロード/処理結果の保存場所
MODEL_SIZE = "medium"
LANGUAGE = "ja"

# --- アプリケーション初期化 ---
app = FastAPI()
os.makedirs(PROCESSING_DIR, exist_ok=True)

# 1. /download エンドポイント (生成された .ans.wav をブラウザに返す)
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")
logger.info(f"'{PROCESSING_DIR}' ディレクトリを /download としてマウントしました。")


# ---------------------------
# バックグラウンド処理関数 (WebSocketオブジェクトを追加)
# ---------------------------
async def process_audio_file(audio_path: str, original_filename: str, websocket: WebSocket):
    """
    アップロードされた音声ファイルを受け取り、一連の処理を実行し、
    完了したらWebSocketでクライアントに通知する
    """
    logger.info(f"[TASK START] ファイル処理開始: {original_filename}")
    question_text = ""
    answer_text = ""
    
    try:
        # --- 1. 文字起こし ---
        output_txt_path = os.path.join(PROCESSING_DIR, original_filename + ".txt")
        logger.info(f"[TASK] (1/4) 文字起こし中...")
        question_text = await asyncio.to_thread(
            whisper_text_only,
            audio_path, language=LANGUAGE, output_txt=output_txt_path
        )
        logger.info(f"[TASK] (1/4) 文字起こし完了: {question_text}")

        # --- 2. 回答生成 (OpenAI) ---
        logger.info(f"[TASK] (2/4) 回答生成中...")
        answer_text = await asyncio.to_thread(generate_answer, question_text)
        logger.info(f"[TASK] (2/4) 回答生成完了: {answer_text[:30]}...")

        # --- 3. 回答の保存 (.txt) ---
        logger.info(f"[TASK] (3/4) 回答テキスト保存中...")
        answer_file_path = os.path.join(PROCESSING_DIR, original_filename + ".ans.txt")
        with open(answer_file_path, "w", encoding="utf-8") as f:
            f.write(answer_text)
        
        # --- 4. 回答の音声合成 (Fish-Speech) ---
        logger.info(f"[TASK] (4/4) 回答の音声合成中...")
        answer_wav_filename = original_filename + ".ans.wav"
        answer_wav_path_abs = os.path.abspath(os.path.join(PROCESSING_DIR, answer_wav_filename))
        
        success_tts = await asyncio.to_thread(
            synthesize_speech,
            text_to_speak=answer_text,
            output_wav_path=answer_wav_path_abs
        )
        
        if success_tts:
            logger.info(f"[TASK] (4/4) 音声合成 完了。クライアントに通知します。")
            # ★ 優先度2 & 3: 完了通知にテキストデータを追加
            download_url = f"/download/{answer_wav_filename}"
            await websocket.send_json({
                "status": "complete",
                "message": "回答の準備ができました。",
                "audio_url": download_url,
                "question_text": question_text, # ★ 優先度3
                "answer_text": answer_text      # ★ 優先度3
            })
        else:
            logger.warning(f"[WARN] (4/4) 音声合成に失敗しました。")
            await websocket.send_json({"status": "error", "message": "音声合成に失敗しました。"})
        
        logger.info(f"[TASK END] 全処理完了: {original_filename}")

    except Exception as e:
        logger.error(f"[TASK ERROR] '{original_filename}' の処理中にエラーが発生しました: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": f"処理中にエラーが発生しました: {e}"})
        except WebSocketDisconnect:
            pass # クライアントが切断済みなら何もしない

# ---------------------------
# WebSocket エンドポイント ( /ws )
# (ブラウザからのマイク音声データ受信)
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] クライアントが接続しました。")
    try:
        while True:
            # 1. ブラウザから音声データ (Blob) を受信
            audio_data = await websocket.receive_bytes()
            
            # 2. 一時ファイルとして保存 (ブラウザからは .webm 形式が多い)
            temp_id = f"ws_{int(time.time())}"
            temp_input_path = os.path.join(PROCESSING_DIR, f"{temp_id}.webm") 
            
            with open(temp_input_path, "wb") as f:
                f.write(audio_data)
            
            # 3. ffmpeg で .wav に変換 (Whisperが処理できる形式へ)
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)
            
            # 16kHz モノラル 16bit PCM に変換
            cmd = [
                "ffmpeg",
                "-i", temp_input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-y", # 常に上書き
                output_wav_path
            ]
            
            logger.info(f"[WS] ffmpeg 変換実行: {temp_input_path} -> {output_wav_path}")
            # 非同期でサブプロセスを実行
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                logger.error(f"[WS ERROR] ffmpeg 変換失敗: {stderr.decode()}")
                await websocket.send_json({"status": "error", "message": "音声形式の変換に失敗しました。"})
                continue
            
            logger.info(f"[WS] ffmpeg 変換成功。")
            
            # 4. バックグラウンド処理タスクを開始 (websocketオブジェクトを渡す)
            asyncio.create_task(process_audio_file(
                output_wav_path, 
                output_wav_filename, # .wav ファイル名を基準にする
                websocket
            ))
            
            # 5. クライアントに「処理中」を通知
            await websocket.send_json({"status": "processing", "message": "文字起こしと回答生成を開始しました..."})

            # 6. 一時入力ファイル (webm) を削除
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    except WebSocketDisconnect:
        logger.info("[WS] クライアントが切断しました。")
    except Exception as e:
        logger.error(f"[WS ERROR] WebSocketエラー: {e}", exc_info=True)
    finally:
        await websocket.close()


# ---------------------------
# ルート ( / )
# (ブラウザにHTML/JavaScriptを返す)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>リアルタイム音声応答</title>
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #f4f4f4; }
            #container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 600px; }
            button { font-size: 1.2rem; padding: 0.8rem 1.5rem; border: none; border-radius: 5px; cursor: pointer; margin: 0.5rem; }
            #recordButton { background: #dc3545; color: white; }
            #recordButton.recording { background: #28a745; }
            #status { margin-top: 1.5rem; font-size: 1.1rem; color: #333; min-height: 2em; }
            
            /* ★ 優先度3: テキスト表示用スタイル */
            #qa-display { 
                margin: 1.5rem auto 0 auto; 
                text-align: left; 
                width: 100%; 
                border-top: 1px solid #eee; 
                padding-top: 1rem; 
            }
            #qa-display div { 
                margin-bottom: 1rem; 
                padding: 0.5rem;
                background: #f9f9f9;
                border-radius: 5px;
                white-space: pre-wrap; /* 改行を反映 */
                word-wrap: break-word; /* 長い単語を折り返す */
            }
            #qa-display div:empty { display: none; } /* 空のときは非表示 */
            #question-text::before { content: '■ あなたの質問:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #007bff;}
            #answer-text::before { content: '■ AIの回答:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #28a745;}

            #audioPlayback { margin-top: 1rem; }
            #audioPlayback audio { width: 100%; }
            
            /* ★ 優先度2: ダウンロードリンク用スタイル */
            #downloadLink { margin-top: 0.5rem; font-size: 0.9rem; }
            
        </style>
    </head>
    <body>
        <div id="container">
            <h1>音声応答システム</h1>
            <p>ボタンを押して話しかけてください。</p>
            <button id="recordButton">録音開始</button>
            <div id="status">ここにステータスが表示されます</div>
            
            <div id="qa-display">
                <div id="question-text"></div>
                <div id="answer-text"></div>
            </div>

            <div id="audioPlayback"></div>
            <div id="downloadLink"></div>
        </div>

        <script>
            const recordButton = document.getElementById('recordButton');
            const statusDiv = document.getElementById('status');
            const audioPlayback = document.getElementById('audioPlayback');
            // ★ 優先度2 & 3: 要素を取得
            const downloadLinkDiv = document.getElementById('downloadLink');
            const questionTextDiv = document.getElementById('question-text');
            const answerTextDiv = document.getElementById('answer-text');

            let ws;
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;

            function connectWebSocket() {
                // RunPodのWebSocket (WSS) URLを自動的に決定
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                const wsUrl = wsProtocol + window.location.host + '/ws';
                
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    console.log('WebSocket 接続成功');
                    statusDiv.textContent = '準備完了。ボタンを押して録音してください。';
                    recordButton.disabled = false;
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log('サーバーからメッセージ:', data);
                    
                    statusDiv.textContent = data.message; // ステータスを更新

                    if (data.status === 'complete' && data.audio_url) {
                        // 回答の音声URLが送られてきたら再生
                        playAudio(data.audio_url);
                        
                        // ★ 優先度3: テキストを表示
                        questionTextDiv.textContent = data.question_text || '（質問を聞き取れませんでした）';
                        answerTextDiv.textContent = data.answer_text || '（回答を生成できませんでした）';
                        
                        // ★ 優先度2: ダウンロードリンクを生成
                        createDownloadLink(data.audio_url);

                        recordButton.disabled = false; // 次の録音を許可
                    } else if (data.status === 'error') {
                        recordButton.disabled = false; // エラー時も録音許可
                        answerTextDiv.textContent = `エラー: ${data.message}`; // ★ 優先度3
                    }
                };

                ws.onclose = () => {
                    console.log('WebSocket 接続切断');
                    statusDiv.textContent = 'サーバーとの接続が切れました。リロードしてください。';
                    recordButton.disabled = true;
                };

                ws.onerror = (error) => {
                    console.error('WebSocket エラー:', error);
                    statusDiv.textContent = '接続エラーが発生しました。';
                };
            }

            async function startRecording() {
                try {
                    // ★ 優先度2 & 3: 以前の結果をクリア
                    clearResults();
                
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    // audio/webm;codecs=opus 形式で録音 (ffmpegで変換可能)
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        // 録音停止時にデータをBlobとして結合
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            // WebSocketでサーバーに送信
                            ws.send(audioBlob);
                            statusDiv.textContent = '音声を送信中... サーバー処理を待っています。';
                        } else {
                            statusDiv.textContent = 'サーバーに接続されていません。';
                        }
                        
                        // ストリームを停止
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    recordButton.textContent = '録音停止';
                    recordButton.classList.add('recording');
                    statusDiv.textContent = '録音中...';

                } catch (err) {
                    console.error('マイクへのアクセスに失敗:', err);
                    statusDiv.textContent = 'マイクへのアクセスが許可されていません。';
                }
            }

            function stopRecording() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    recordButton.textContent = '録音開始';
                    recordButton.classList.remove('recording');
                    recordButton.disabled = true; // 処理完了まで押せないようにする
                }
            }
            
            // ★ 優先度2 & 3: 結果をクリアする関数
            function clearResults() {
                audioPlayback.innerHTML = ''; // 古いオーディオをクリア
                downloadLinkDiv.innerHTML = ''; // 古いリンクをクリア
                questionTextDiv.textContent = ''; // 古い質問をクリア
                answerTextDiv.textContent = ''; // 古い回答をクリア
            }

            function playAudio(url) {
                audioPlayback.innerHTML = ''; // 古いオーディオをクリア
                const audio = new Audio(url);
                audio.controls = true;
                audio.autoplay = true; // 自動再生
                audioPlayback.appendChild(audio);
            }
            
            // ★ 優先度2: ダウンロードリンクを作成する関数
            function createDownloadLink(url) {
                downloadLinkDiv.innerHTML = ''; // クリア
                const a = document.createElement('a');
                a.href = url;
                a.textContent = '回答音声をダウンロード';
                // URLからファイル名 (ws_....wav) を抽出してdownload属性に設定
                a.download = url.split('/').pop(); 
                downloadLinkDiv.appendChild(a);
            }

            recordButton.onclick = () => {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            };

            // ページ読み込み時にWebSocket接続を開始
            window.onload = () => {
                recordButton.disabled = true; // 接続完了まで押せないように
                connectWebSocket();
            };

        </script>
    </body>
    </html>
    """

# ---------------------------
# サーバー起動
# ---------------------------
if __name__ == "__main__":
    # RunPodのデフォルトHTTPポート (8000など) に合わせる
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    # ★ 優先度1: Uvicornのロギングをカスタムロガーに合わせる
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)