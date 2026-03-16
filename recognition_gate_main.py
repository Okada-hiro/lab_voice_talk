import asyncio
from dataclasses import dataclass

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import parallel_faster_main as base


app = FastAPI()


@dataclass
class ActivationState:
    active: bool = False
    person_id: str | None = None


STATE = ActivationState()
STATE_LOCK = asyncio.Lock()
WS_CLIENTS: set[WebSocket] = set()
WS_CLIENTS_LOCK = asyncio.Lock()


class ApproachPayload(BaseModel):
    person_id: str | None = None


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


async def _speak_text_to_websocket(websocket: WebSocket, text: str) -> None:
    pcm_bytes = await asyncio.to_thread(base.synthesize_speech_to_memory, text)
    if not pcm_bytes:
        return
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
    await websocket.send_json({"status": "audio_sentence_done", "sentence_id": 1, "last_chunk_id": 1, "total_bytes": len(pcm_bytes)})
    await websocket.send_json({"status": "complete", "answer_text": text})


async def _broadcast_greeting(person_id: str | None) -> None:
    text = f"{person_id}さん、こんにちは。" if person_id else "こんにちは。"
    async with WS_CLIENTS_LOCK:
        clients = list(WS_CLIENTS)
    for websocket in clients:
        try:
            await _speak_text_to_websocket(websocket, text)
        except Exception:
            pass


@app.post("/recognition/approach")
async def recognition_approach(payload: ApproachPayload) -> dict[str, object]:
    async with STATE_LOCK:
        was_active = STATE.active
        STATE.active = True
        STATE.person_id = payload.person_id
    await _broadcast_json(
        {
            "status": "system_info",
            "message": f"認識システムが接近を検知しました。person_id={payload.person_id or 'unknown'}",
        }
    )
    if not was_active:
        await _broadcast_greeting(payload.person_id)
    return {"ok": True, "active": True, "person_id": payload.person_id}


@app.post("/recognition/leave")
async def recognition_leave(payload: ApproachPayload) -> dict[str, object]:
    async with STATE_LOCK:
        STATE.active = False
        STATE.person_id = payload.person_id
    await _broadcast_json(
        {
            "status": "system_info",
            "message": f"認識システムが離脱を検知しました。person_id={payload.person_id or 'unknown'}",
        }
    )
    return {"ok": True, "active": False, "person_id": payload.person_id}


@app.get("/recognition/state")
async def recognition_state() -> dict[str, object]:
    async with STATE_LOCK:
        return {"active": STATE.active, "person_id": STATE.person_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    base.logger.info("[WS] Client Connected (recognition gate).")
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


@app.get("/", response_class=HTMLResponse)
async def root():
    html = await base.get_root()
    html = html.replace("Team Chat AI", "Recognition Gate Chat AI", 1)
    html = html.replace("接続待機中...", "認識システムからの接近待ち...", 1)
    return HTMLResponse(html)


if __name__ == "__main__":
    port = int(base.os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
