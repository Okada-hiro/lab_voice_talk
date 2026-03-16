from __future__ import annotations

import asyncio
import html
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import parallel_faster_main as base

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ultralytics_root = REPO_ROOT / "ultralytics"
if str(ultralytics_root) not in sys.path:
    sys.path.insert(0, str(ultralytics_root))

from recognition.config import AppConfig
from recognition.pipeline import ReceptionMonitor


PORT = int(os.getenv("PORT", "8000"))
LIVE_PERSON_MODEL = os.getenv("RECEPTION_PERSON_MODEL", "yolo11n.pt")
LIVE_DEVICE = os.getenv("RECEPTION_DEVICE", "auto")
LIVE_DATABASE_DIR = Path(os.getenv("RECEPTION_DATABASE_DIR", REPO_ROOT / "data_base")).resolve()
LIVE_JPEG_QUALITY = int(os.getenv("RECEPTION_JPEG_QUALITY", "85"))
SCREEN_ROOT = REPO_ROOT / "application" / "screen"

app = FastAPI(title="Reception Assistant", version="1.0.0")
if SCREEN_ROOT.exists():
    app.mount("/app-assets", StaticFiles(directory=str(SCREEN_ROOT)), name="app-assets")


@dataclass
class ActivationState:
    active: bool = False
    person_id: str | None = None


STATE = ActivationState()
STATE_LOCK = asyncio.Lock()
WS_CLIENTS: set[WebSocket] = set()
WS_CLIENTS_LOCK = asyncio.Lock()

_live_monitor: ReceptionMonitor | None = None
_live_monitor_lock = threading.Lock()
_live_frame_index = 0
_greeting_pcm: bytes | None = None
_personal_greeting_cache: dict[str, bytes] = {}


def _render_combined_page() -> str:
    return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reception Assistant</title>
  <style>
    :root {
      --bg: #f6f2e8;
      --panel: #fffaf0;
      --ink: #1c1a18;
      --line: #d7c8a8;
      --link: #0d5c63;
    }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, var(--bg), #efe5d1);
      color: var(--ink);
    }
    main {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(60, 40, 10, 0.08);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 1200px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
    iframe {
      width: 100%;
      min-height: 920px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }
    a {
      color: var(--link);
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .note {
      color: #6b665d;
    }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>Reception Assistant</h1>
      <p class="note">
        左: 人物検出・顔検出・顔認識。右: 音声会話 UI。人物が近づくまでは会話は待機します。
      </p>
      <div class="grid">
        <section>
          <h2>Vision</h2>
          <iframe src="/live" allow="camera"></iframe>
        </section>
        <section>
          <h2>Voice</h2>
          <iframe src="/voice-ui" allow="microphone; autoplay"></iframe>
        </section>
      </div>
    </div>
  </main>
</body>
</html>"""


def _load_screen_html(filename: str) -> str:
    html_path = SCREEN_ROOT / filename
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"screen file not found: {filename}")
    html_text = html_path.read_text(encoding="utf-8")
    html_text = html_text.replace("./styles.css", "/app-assets/styles.css")
    html_text = html_text.replace("./app.js", "/app-assets/app.js")
    html_text = html_text.replace("./live.js", "/app-assets/live.js")
    return html_text


def _render_live_page() -> str:
    return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Recognition</title>
  <style>
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: #fffaf0;
      color: #1c1a18;
    }
    main {
      padding: 16px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 16px;
    }
    button, input {
      font: inherit;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      background: #9a673a;
      color: white;
      cursor: pointer;
    }
    button.secondary {
      background: #d7c8a8;
      color: #1c1a18;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
    video, img {
      width: 100%;
      border-radius: 12px;
      background: #ddd;
    }
    .status {
      margin: 8px 0 16px;
      min-height: 3em;
      white-space: pre-line;
    }
  </style>
</head>
<body>
  <main>
    <div class="controls">
      <button id="start">Start Camera</button>
      <button id="stop" class="secondary">Stop</button>
      <label>Base Interval (ms) <input id="interval" type="number" min="150" step="50" value="800"></label>
      <label>Width <input id="width" type="number" min="320" step="80" value="960"></label>
    </div>
    <div id="status" class="status">待機中</div>
    <div class="grid">
      <section>
        <h2>Camera</h2>
        <video id="video" autoplay playsinline muted></video>
      </section>
      <section>
        <h2>Processed</h2>
        <img id="processed" alt="processed frame">
      </section>
    </div>
  </main>
  <script>
    const video = document.getElementById("video");
    const processed = document.getElementById("processed");
    const statusEl = document.getElementById("status");
    const startBtn = document.getElementById("start");
    const stopBtn = document.getElementById("stop");
    const intervalInput = document.getElementById("interval");
    const widthInput = document.getElementById("width");

    let stream = null;
    let timer = null;
    let frameCount = 0;
    let busy = false;

    function computeNextInterval(personCount, faceCount) {
      const baseInterval = Math.max(150, Number(intervalInput.value || 800));
      if (faceCount > 0) return Math.min(baseInterval, 300);
      if (personCount > 0) return Math.min(baseInterval, 500);
      return Math.max(baseInterval, 800);
    }

    function scheduleNextCapture(delayMs) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      timer = setTimeout(captureAndSend, delayMs);
    }

    function stopCamera() {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
        stream = null;
      }
      busy = false;
      statusEl.textContent = "停止しました";
    }

    async function captureAndSend() {
      if (!stream || busy) return;
      busy = true;
      try {
        const width = Number(widthInput.value || 960);
        const scale = width / video.videoWidth;
        const height = Math.max(1, Math.round(video.videoHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, width, height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
        const form = new FormData();
        form.append("frame", blob, "frame.jpg");
        const response = await fetch("/api/live-frame", { method: "POST", body: form });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text);
        }
        const eventsJson = response.headers.get("x-track-events") || "[]";
        const events = JSON.parse(eventsJson);
        const personCount = Number(response.headers.get("x-person-count") || "0");
        const faceCount = Number(response.headers.get("x-face-count") || "0");
        const imageBlob = await response.blob();
        processed.src = URL.createObjectURL(imageBlob);
        frameCount += 1;
        const nextInterval = computeNextInterval(personCount, faceCount);
        const lines = [`Processed frame ${frameCount}.`];
        lines.push(`next interval: ${nextInterval} ms (persons=${personCount}, faces=${faceCount})`);
        if (events.length) {
          for (const event of events) {
            lines.push(`${event.event_type} track=${event.track_id}${event.person_id ? " " + event.person_id : ""}`);
          }
        } else {
          lines.push("No approach/leave events.");
        }
        statusEl.textContent = lines.join("\\n");
        scheduleNextCapture(nextInterval);
      } catch (error) {
        statusEl.textContent = `Error: ${error}`;
        scheduleNextCapture(Math.max(500, Number(intervalInput.value || 800)));
      } finally {
        busy = false;
      }
    }

    async function startCamera() {
      stopCamera();
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      video.srcObject = stream;
      await video.play();
      statusEl.textContent = "カメラ開始";
      scheduleNextCapture(0);
    }

    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);
  </script>
</body>
</html>"""


def _get_live_monitor() -> ReceptionMonitor:
    global _live_monitor
    if _live_monitor is None:
        config = AppConfig(
            person_model=LIVE_PERSON_MODEL,
            device=LIVE_DEVICE,
            database_dir=LIVE_DATABASE_DIR,
            save_snapshots=False,
        )
        _live_monitor = ReceptionMonitor(config)
    return _live_monitor


async def _broadcast_json(payload: dict) -> None:
    async with WS_CLIENTS_LOCK:
        clients = list(WS_CLIENTS)
    stale_clients: list[WebSocket] = []
    for websocket in clients:
        try:
            await websocket.send_json(payload)
        except Exception:
            stale_clients.append(websocket)
    if stale_clients:
        async with WS_CLIENTS_LOCK:
            for websocket in stale_clients:
                WS_CLIENTS.discard(websocket)


async def _broadcast_pcm(pcm_bytes: bytes, text: str) -> None:
    async with WS_CLIENTS_LOCK:
        clients = list(WS_CLIENTS)
    stale_clients: list[WebSocket] = []
    for websocket in clients:
        try:
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text})
            await websocket.send_json(
                {
                    "status": "audio_chunk_meta",
                    "sentence_id": 1,
                    "chunk_id": 1,
                    "global_chunk_id": 1,
                    "arrival_seq": 1,
                    "byte_len": len(pcm_bytes),
                    "sample_rate": 16000,
                }
            )
            await websocket.send_bytes(pcm_bytes)
            await websocket.send_json(
                {"status": "audio_sentence_done", "sentence_id": 1, "last_chunk_id": 1, "total_bytes": len(pcm_bytes)}
            )
            await websocket.send_json({"status": "complete", "answer_text": text})
        except Exception:
            stale_clients.append(websocket)
    if stale_clients:
        async with WS_CLIENTS_LOCK:
            for websocket in stale_clients:
                WS_CLIENTS.discard(websocket)


async def _ensure_greeting_pcm() -> bytes | None:
    global _greeting_pcm
    if _greeting_pcm is None:
        try:
            _greeting_pcm = await asyncio.to_thread(base.synthesize_speech_to_memory, "こんにちは。受付です。")
        except Exception as exc:
            base.logger.error(f"[GREETING] failed to precompute greeting: {exc}", exc_info=True)
            _greeting_pcm = None
    return _greeting_pcm


def _build_greeting_text(person_id: str | None) -> str:
    if person_id:
        return f"こんにちは。{person_id}さん。"
    return "こんにちは。受付です。"


async def _get_greeting_pcm(person_id: str | None) -> tuple[bytes | None, str]:
    text = _build_greeting_text(person_id)
    if not person_id:
        return await _ensure_greeting_pcm(), text
    cached = _personal_greeting_cache.get(person_id)
    if cached is not None:
        return cached, text
    try:
        pcm = await asyncio.to_thread(base.synthesize_speech_to_memory, text)
    except Exception as exc:
        base.logger.error(f"[GREETING] failed to synthesize personal greeting for {person_id}: {exc}", exc_info=True)
        return await _ensure_greeting_pcm(), "こんにちは。受付です。"
    if pcm:
        _personal_greeting_cache[person_id] = pcm
    return pcm, text


async def _handle_track_events(track_events) -> None:
    for track_event in track_events:
        if track_event.event_type == "approached":
            should_greet = False
            async with STATE_LOCK:
                if not STATE.active:
                    STATE.active = True
                    STATE.person_id = track_event.person_id
                    should_greet = True
            await _broadcast_json(
                {
                    "status": "system_info",
                    "message": f"接近を検知しました。{track_event.person_id or 'guest'} さん、どうぞお話しください。",
                }
            )
            if should_greet:
                pcm, greeting_text = await _get_greeting_pcm(track_event.person_id)
                if pcm:
                    await _broadcast_pcm(pcm, greeting_text)
        elif track_event.event_type == "left":
            async with STATE_LOCK:
                STATE.active = False
                STATE.person_id = track_event.person_id
            await _broadcast_json(
                {
                    "status": "system_info",
                    "message": f"離脱を検知しました。{track_event.person_id or 'guest'} さん、ありがとうございました。",
                }
            )


@app.on_event("startup")
async def startup_event() -> None:
    await _ensure_greeting_pcm()


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_render_combined_page())


@app.get("/live", response_class=HTMLResponse)
async def live_page() -> HTMLResponse:
    return HTMLResponse(_render_live_page())


@app.get("/voice-ui", response_class=HTMLResponse)
async def voice_ui() -> HTMLResponse:
    html_text = await base.get_root()
    html_text = html_text.replace("Team Chat AI", "Reception Voice", 1)
    html_text = html_text.replace("接続待機中...", "認識待機中...", 1)
    return HTMLResponse(html_text)


@app.get("/app", response_class=HTMLResponse)
async def screen_live_app() -> HTMLResponse:
    return HTMLResponse(_load_screen_html("live.html"))


@app.get("/app/manual", response_class=HTMLResponse)
async def screen_manual_app() -> HTMLResponse:
    return HTMLResponse(_load_screen_html("index.html"))


@app.post("/enable-registration")
async def enable_registration():
    return await base.enable_registration()


@app.post("/api/live-frame")
async def live_frame(frame: UploadFile = File(...)) -> Response:
    global _live_frame_index

    data = await frame.read()
    np_buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded frame.")

    with _live_monitor_lock:
        monitor = _get_live_monitor()
        frame_index = _live_frame_index
        _live_frame_index += 1
        annotated, event = monitor.process_frame(image, frame_index)

    if event.track_events:
        await _handle_track_events(event.track_events)

    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated frame.")

    headers = {
        "x-track-events": json.dumps(
            [
                {"track_id": item.track_id, "event_type": item.event_type, "person_id": item.person_id}
                for item in event.track_events
            ],
            ensure_ascii=False,
        ),
        "x-person-count": str(len(event.persons)),
        "x-face-count": str(len(event.faces)),
        "x-primary-person-id": next(
            (
                item.person_id
                for item in event.matches
                if item.person_id
            ),
            "",
        ),
    }
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers=headers)


@app.get("/recognition/state")
async def recognition_state() -> dict[str, object]:
    async with STATE_LOCK:
        return {"active": STATE.active, "person_id": STATE.person_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    base.logger.info("[WS] Client Connected (reception main).")
    async with WS_CLIENTS_LOCK:
        WS_CLIENTS.add(websocket)

    vad_iterator = base.VADIterator(
        base.vad_model,
        threshold=0.95,
        sampling_rate=16000,
        min_silence_duration_ms=200,
        speech_pad_ms=50,
    )

    audio_buffer = []
    is_speaking = False
    interruption_triggered = False

    window_size_samples = 512
    sample_rate = 16000
    check_speaker_samples = 30000
    chat_history = []

    try:
        await websocket.send_json({"status": "system_info", "message": "認識システムからの接近待ちです。"})
        while True:
            data_bytes = await websocket.receive_bytes()
            async with STATE_LOCK:
                current_active = STATE.active
            if not current_active:
                continue

            audio_chunk_np = np.frombuffer(data_bytes, dtype=np.float32).copy()
            offset = 0
            while offset + window_size_samples <= len(audio_chunk_np):
                window_np = audio_chunk_np[offset : offset + window_size_samples]
                offset += window_size_samples
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(base.DEVICE)

                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)

                if speech_dict:
                    if "start" in speech_dict:
                        base.logger.info("🗣️ Speech START")
                        is_speaking = True
                        interruption_triggered = False
                        audio_buffer = [window_np]
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                    elif "end" in speech_dict:
                        base.logger.info("🤫 Speech END")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            full_audio = np.concatenate(audio_buffer)

                            if len(full_audio) / sample_rate < 0.2:
                                base.logger.info("Noise detected")
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                await base.process_voice_pipeline(full_audio, websocket, chat_history)
                            audio_buffer = []
                else:
                    if is_speaking:
                        audio_buffer.append(window_np)
                        current_len = sum(len(c) for c in audio_buffer)
                        if not interruption_triggered and not base.NEXT_AUDIO_IS_REGISTRATION and current_len > check_speaker_samples:
                            temp_audio = np.concatenate(audio_buffer)
                            temp_tensor = torch.from_numpy(temp_audio).float().unsqueeze(0)
                            is_verified, spk_id = await asyncio.to_thread(base.speaker_guard.identify_speaker, temp_tensor)
                            if is_verified:
                                base.logger.info(f"⚡ [Barge-in] {spk_id} の声を検知！停止指示。")
                                await websocket.send_json({"status": "interrupt", "message": "🛑 音声停止"})
                                interruption_triggered = True

    except WebSocketDisconnect:
        base.logger.info("[WS] Disconnected")
    except Exception as exc:
        base.logger.error(f"[WS ERROR] {exc}", exc_info=True)
    finally:
        vad_iterator.reset_states()
        async with WS_CLIENTS_LOCK:
            WS_CLIENTS.discard(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
