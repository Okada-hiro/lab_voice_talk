#今はこれ! 2月20日 一旦安定する

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import asyncio
import logging
import sys
import os
import io
import re
import time

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 必要なモジュールのインポート ---
try:
    from transcribe_func import GLOBAL_ASR_MODEL_INSTANCE
    from new_answer_generator import generate_answer_stream
    import new_text_to_speech as tts_module
    from new_text_to_speech import (
        synthesize_speech,
        synthesize_speech_to_memory,
        synthesize_speech_to_memory_stream,
    )
    from new_speaker_filter import SpeakerGuard
except ImportError as e:
    logger.error(f"[ERROR] 必要なモジュールが見つかりません: {e}")
    sys.exit(1)

# --- グローバル設定 ---
PROCESSING_DIR = "incoming_audio"
os.makedirs(PROCESSING_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using Device: {DEVICE}")

app = FastAPI()
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")

# SpeakerGuard初期化
speaker_guard = SpeakerGuard()
NEXT_AUDIO_IS_REGISTRATION = False

# --- Silero VAD のロード ---
logger.info("⏳ Loading Silero VAD model...")
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_model.to(DEVICE)
    logger.info("✅ Silero VAD model loaded.")
except Exception as e:
    logger.critical(f"Silero VAD Load Failed: {e}")
    sys.exit(1)


# --- API: 登録モード切替 ---
@app.post("/enable-registration")
async def enable_registration():
    global NEXT_AUDIO_IS_REGISTRATION
    NEXT_AUDIO_IS_REGISTRATION = True
    logger.info("【モード切替】次の発話を新規話者として登録します")
    return {"message": "登録モード待機中"}


# --- ヘルパー: 音声処理パイプライン ---
async def process_voice_pipeline(audio_float32_np, websocket: WebSocket, chat_history: list):
    global NEXT_AUDIO_IS_REGISTRATION

    # --- ★追加: 自分の声を保存して確認できるようにする ---
    import soundfile as sf
    # 毎回上書きされます
    debug_path = f"{PROCESSING_DIR}/last_user_input.wav"
    sf.write(debug_path, audio_float32_np, 16000)
    logger.info(f"🎤 [DEBUG] あなたの声を保存しました: {debug_path}")
    # --------------------------------------------------

    # SpeakerGuard用に Tensor化
    voice_tensor = torch.from_numpy(audio_float32_np).float().unsqueeze(0)
    
    # SpeakerGuard用に Tensor化
    voice_tensor = torch.from_numpy(audio_float32_np).float().unsqueeze(0)
    
    speaker_id = "Unknown"
    is_allowed = False

    # ---------------------------
    # 1. 話者判定 / 登録ロジック
    # ---------------------------
    if NEXT_AUDIO_IS_REGISTRATION:
        temp_reg_path = f"{PROCESSING_DIR}/reg_{id(audio_float32_np)}.wav"
        import soundfile as sf
        sf.write(temp_reg_path, audio_float32_np, 16000)
        
        new_id = await asyncio.to_thread(speaker_guard.register_new_speaker, temp_reg_path)
        NEXT_AUDIO_IS_REGISTRATION = False 
        
        if new_id:
            speaker_id = new_id
            is_allowed = True
            await websocket.send_json({"status": "system_info", "message": f"✅ {new_id} を登録しました！会話を続けます。"})
        else:
            await websocket.send_json({"status": "error", "message": "登録に失敗しました"})
            return
            
    else:
        is_allowed, detected_id = await asyncio.to_thread(speaker_guard.identify_speaker, voice_tensor)
        speaker_id = detected_id

    # ---------------------------
    #  2. アクセス制御
    if not is_allowed:
        # ★★★ ここを修正: 短い音声の誤検知対策 ★★★
        # 音声の長さを秒単位で計算 (サンプル数 / サンプリングレート)
        duration_sec = len(audio_float32_np) / 16000
        
        # 1.0秒未満で認証失敗した場合は、ノイズや短い相槌の可能性が高いため、
        # 警告を出さずに「無視」する。
        if duration_sec < 2.5:
            logger.info(f"[Ignored] Short audio ({duration_sec:.2f}s) failed auth. Treating as noise.")
            await websocket.send_json({"status": "ignored", "message": "..."})
            return

        logger.info("[Access Denied] 登録されていない話者です。")
        await websocket.send_json({
            "status": "system_alert", 
            "message": "⚠️ 外部の会話(未登録)を検知しました。ユーザーとして追加する場合は「メンバー追加」から行ってください。",
            "alert_type": "unregistered" 
        })
        return

    # ---------------------------
    # 3. Whisper 文字起こし
    # ---------------------------
    try:
        if GLOBAL_ASR_MODEL_INSTANCE is None:
            raise ValueError("Whisper Model not loaded")

        logger.info("[TASK] 文字起こし開始")
        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            logger.info("[TASK] 空の認識結果")
            return

        text_with_context = f"【{speaker_id}】 {text}"
        logger.info(f"[TASK] {text_with_context}")

        await websocket.send_json({
            "status": "transcribed",
            "question_text": text,
            "speaker_id": speaker_id 
        })

        # ---------------------------
        # 4. LLM & TTS ストリーミング
        # ---------------------------
        await handle_llm_tts(text_with_context, websocket, chat_history)

    except Exception as e:
        logger.error(f"Pipeline Error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "処理エラー"})


# --- ヘルパー: 回答生成と音声合成 ---
async def handle_llm_tts(text_for_llm: str, websocket: WebSocket, chat_history: list):
    text_buffer = ""
    sentence_count = 0
    full_answer = ""
    split_pattern = r'(?<=[。！？\n])'
    llm_tts_start = time.perf_counter()
    TTS_WORKER_COUNT = 1
    TTS_PREFETCH_AHEAD = 1
    STREAM_EMIT_EVERY_FRAMES = int(os.getenv("PERM_EMIT_EVERY_FRAMES", "4"))
    STREAM_DECODE_WINDOW_FRAMES = int(os.getenv("PERM_DECODE_WINDOW_FRAMES", "80"))
    DETAILED_TIMING = os.getenv("PERM_DETAILED_TIMING", "0") == "1"
    stream_cfg = getattr(tts_module, "DEFAULT_STREAM_PARAMS", {})
    if isinstance(stream_cfg, dict):
        stream_cfg["emit_every_frames"] = STREAM_EMIT_EVERY_FRAMES
        stream_cfg["decode_window_frames"] = STREAM_DECODE_WINDOW_FRAMES
    logger.info(
        "[TTS_CONFIG] "
        f"emit_every_frames={stream_cfg.get('emit_every_frames')} "
        f"decode_window_frames={stream_cfg.get('decode_window_frames')} "
        f"overlap_samples={stream_cfg.get('overlap_samples')} "
        f"first_chunk_emit_every={stream_cfg.get('first_chunk_emit_every')} "
        f"first_chunk_decode_window={stream_cfg.get('first_chunk_decode_window')} "
        f"first_chunk_frames={stream_cfg.get('first_chunk_frames')} "
        f"repetition_penalty={stream_cfg.get('repetition_penalty')} "
        f"repetition_penalty_window={stream_cfg.get('repetition_penalty_window')} "
        f"tts_workers={TTS_WORKER_COUNT} prefetch_ahead={TTS_PREFETCH_AHEAD}"
    )
    logger.info(
        f"[LLM_TTS_FLOW] start text_for_llm_len={len(text_for_llm)} "
        f"history_len={len(chat_history)} split_pattern={split_pattern}"
    )

    iterator = generate_answer_stream(text_for_llm, history=chat_history)

    # 16kHz / PCM16 / mono を維持しつつ、20ms単位で細かく送る
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2
    FRAME_MS = 20
    CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * FRAME_MS // 1000
    STOP = object()
    text_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()
    sentence_enqueued_at = {}
    first_audio_sent_at = None
    worker_stop_count = 0

    def _next_stream_chunk_or_none(gen):
        try:
            return next(gen)
        except StopIteration:
            return None

    def _noop():
        return None

    async def tts_worker(worker_id: int):
        nonlocal worker_stop_count
        logger.info(f"[TTS_WORKER] worker={worker_id} started")
        while True:
            item = await text_queue.get()
            try:
                if item is STOP:
                    worker_stop_count += 1
                    logger.info(
                        f"[TTS_WORKER] worker={worker_id} got_stop "
                        f"worker_stop_count={worker_stop_count}/{TTS_WORKER_COUNT}"
                    )
                    if worker_stop_count == TTS_WORKER_COUNT:
                        await audio_queue.put({"type": "stop"})
                        logger.info("[TTS_WORKER] all workers stopped -> audio_queue stop queued")
                    return

                idx, phrase = item
                sentence_start = time.perf_counter()
                queue_wait_ms = (sentence_start - sentence_enqueued_at.get(idx, sentence_start)) * 1000.0
                logger.info(
                    f"[TTS_TIMING] worker={worker_id} sentence={idx} "
                    f"stage=tts_start text_len={len(phrase)} queue_wait_ms={queue_wait_ms:.1f}"
                )
                total_len = 0
                tts_chunk_count = 0
                first_chunk_ready_ms = None
                try:
                    stream_gen = synthesize_speech_to_memory_stream(phrase)
                    while True:
                        # (E) to_thread / スケジューリング固定費の近似
                        sched_probe_start = time.perf_counter()
                        await asyncio.to_thread(_noop)
                        to_thread_overhead_ms = (time.perf_counter() - sched_probe_start) * 1000.0

                        chunk_wait_start = time.perf_counter()
                        pcm_chunk = await asyncio.to_thread(_next_stream_chunk_or_none, stream_gen)
                        chunk_gen_ms = (time.perf_counter() - chunk_wait_start) * 1000.0
                        approx_model_ms = max(0.0, chunk_gen_ms - to_thread_overhead_ms)
                        if pcm_chunk is None:
                            break
                        tts_chunk_count += 1
                        if first_chunk_ready_ms is None:
                            first_chunk_ready_ms = (time.perf_counter() - sentence_start) * 1000.0
                        total_len += len(pcm_chunk)
                        created_at = time.perf_counter()
                        await audio_queue.put(
                            {
                                "type": "chunk",
                                "sentence_idx": idx,
                                "tts_chunk_idx": tts_chunk_count,
                                "worker_id": worker_id,
                                "audio_bytes": pcm_chunk,
                                "total_bytes": 0,
                                "chunk_gen_ms": chunk_gen_ms,
                                "created_at": created_at,
                            }
                        )
                        logger.info(
                            f"[TTS_TIMING] worker={worker_id} sentence={idx} tts_chunk={tts_chunk_count} "
                            f"chunk_bytes={len(pcm_chunk)} chunk_gen_ms={chunk_gen_ms:.1f}"
                        )
                        if DETAILED_TIMING:
                            logger.info(
                                f"[TTS_DETAILED] worker={worker_id} sentence={idx} chunk={tts_chunk_count} "
                                f"to_thread_overhead_ms={to_thread_overhead_ms:.3f} "
                                f"approx_model_ms={approx_model_ms:.1f}"
                            )
                except Exception as e:
                    logger.error(
                        f"[TTS_TIMING] worker={worker_id} sentence={idx} stream_tts_failed: {e}",
                        exc_info=True,
                    )

                # ストリーミングで1チャンクも取れなかった場合は非ストリーミングへフォールバック
                if tts_chunk_count == 0:
                    logger.warning(
                        f"[TTS_TIMING] worker={worker_id} sentence={idx} no_stream_chunk -> fallback_non_streaming"
                    )
                    fallback_start = time.perf_counter()
                    pcm_all = await asyncio.to_thread(synthesize_speech_to_memory, phrase)
                    fallback_ms = (time.perf_counter() - fallback_start) * 1000.0
                    if pcm_all:
                        tts_chunk_count = 1
                        total_len = len(pcm_all)
                        if first_chunk_ready_ms is None:
                            first_chunk_ready_ms = (time.perf_counter() - sentence_start) * 1000.0
                        await audio_queue.put(
                            {
                                "type": "chunk",
                                "sentence_idx": idx,
                                "tts_chunk_idx": tts_chunk_count,
                                "worker_id": worker_id,
                                "audio_bytes": pcm_all,
                                "total_bytes": 0,
                                "chunk_gen_ms": fallback_ms,
                                "created_at": time.perf_counter(),
                            }
                        )
                        logger.info(
                            f"[TTS_TIMING] worker={worker_id} sentence={idx} fallback_chunk_bytes={len(pcm_all)} "
                            f"fallback_gen_ms={fallback_ms:.1f}"
                        )
                    else:
                        logger.warning(
                            f"[TTS_TIMING] worker={worker_id} sentence={idx} fallback returned empty pcm"
                        )

                total_tts_ms = (time.perf_counter() - sentence_start) * 1000.0
                await audio_queue.put(
                    {
                        "type": "done",
                        "sentence_idx": idx,
                        "tts_chunk_idx": tts_chunk_count,
                        "worker_id": worker_id,
                        "audio_bytes": b"",
                        "total_bytes": total_len,
                        "chunk_gen_ms": None,
                        "created_at": time.perf_counter(),
                        "first_chunk_ready_ms": first_chunk_ready_ms,
                        "total_tts_ms": total_tts_ms,
                        "queue_wait_ms": queue_wait_ms,
                    }
                )
                logger.info(
                    f"[TTS_TIMING] worker={worker_id} sentence={idx} stage=tts_done "
                    f"tts_chunk_count={tts_chunk_count} total_bytes={total_len} total_tts_ms={total_tts_ms:.1f}"
                )
            finally:
                text_queue.task_done()

    async def audio_sender_worker():
        nonlocal first_audio_sent_at
        sent_arrival_seq = 0
        logger.info("[AUDIO_SENDER] started")
        while True:
            item = await audio_queue.get()
            try:
                if item.get("type") == "stop":
                    logger.info("[AUDIO_SENDER] got_stop -> exit")
                    return

                if item["type"] == "chunk":
                    idx = item["sentence_idx"]
                    tts_chunk_idx = item["tts_chunk_idx"]
                    audio_bytes = item["audio_bytes"]
                    send_start = time.perf_counter()
                    queue_to_send_ms = (send_start - item["created_at"]) * 1000.0
                    sent_arrival_seq += 1
                    await websocket.send_json(
                        {
                            "status": "audio_chunk_meta",
                            "sentence_id": idx,
                            "chunk_id": tts_chunk_idx,
                            "arrival_seq": sent_arrival_seq,
                            "byte_len": len(audio_bytes),
                            "sample_rate": SAMPLE_RATE,
                        }
                    )
                    await websocket.send_bytes(audio_bytes)
                    send_ms = (time.perf_counter() - send_start) * 1000.0

                    if first_audio_sent_at is None:
                        first_audio_sent_at = time.perf_counter()
                        first_audio_ms = (first_audio_sent_at - llm_tts_start) * 1000.0
                        logger.info(f"[TTS_TIMING] first_audio_sent_ms={first_audio_ms:.1f}")

                    logger.info(
                        f"[TTS_TIMING] worker={item['worker_id']} sentence={idx} "
                        f"tts_chunk={tts_chunk_idx} stage=send chunk_bytes={len(audio_bytes)} "
                        f"ws_chunks=1 queue_to_send_ms={queue_to_send_ms:.1f} send_ms={send_ms:.1f}"
                    )
                elif item["type"] == "done":
                    await websocket.send_json(
                        {
                            "status": "audio_sentence_done",
                            "sentence_id": item["sentence_idx"],
                            "last_chunk_id": item.get("tts_chunk_idx", 0),
                            "total_bytes": item.get("total_bytes", 0),
                        }
                    )
                    logger.info(
                        f"🚀 Streamed audio {item['sentence_idx']} (Total: {item.get('total_bytes', 0)} bytes) "
                        f"[TTS_TIMING] queue_wait_ms={item.get('queue_wait_ms', 0.0):.1f} "
                        f"first_chunk_ready_ms={item.get('first_chunk_ready_ms', 0.0) or 0.0:.1f} "
                        f"total_tts_ms={item.get('total_tts_ms', 0.0):.1f} "
                        f"tts_chunk_count={item.get('tts_chunk_idx', 0)}"
                    )
            finally:
                audio_queue.task_done()

    tts_tasks = [asyncio.create_task(tts_worker(i + 1)) for i in range(TTS_WORKER_COUNT)]
    sender_task = asyncio.create_task(audio_sender_worker())

    try:
        for chunk in iterator:
            text_buffer += chunk
            full_answer += chunk
            logger.info(
                f"[LLM_STREAM] got_chunk len={len(chunk)} "
                f"buffer_len={len(text_buffer)} full_len={len(full_answer)}"
            )
            
            # ★ "irrelevant" タイプとして送信
            if full_answer.strip() == "[SILENCE]":
                await websocket.send_json({
                    "status": "system_alert", 
                    "message": "⚠️ 会話外の音声と判断しました。会話を続けてください。",
                    "alert_type": "irrelevant"
                })
                return

            sentences = re.split(split_pattern, text_buffer)
            if len(sentences) > 1:
                logger.info(
                    f"[LLM_STREAM] sentence_split count={len(sentences)-1} "
                    f"tail_len={len(sentences[-1])}"
                )
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        await websocket.send_json({"status": "reply_chunk", "text_chunk": sent})
                        sentence_enqueued_at[sentence_count] = time.perf_counter()
                        await text_queue.put((sentence_count, sent))
                        logger.info(
                            f"[LLM_STREAM] enqueued sentence={sentence_count} len={len(sent)} "
                            f"text_queue_size={text_queue.qsize()}"
                        )
                text_buffer = sentences[-1]
        
        if text_buffer.strip():
            sentence_count += 1
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text_buffer})
            sentence_enqueued_at[sentence_count] = time.perf_counter()
            await text_queue.put((sentence_count, text_buffer))
            logger.info(
                f"[LLM_STREAM] enqueued tail sentence={sentence_count} len={len(text_buffer)} "
                f"text_queue_size={text_queue.qsize()}"
            )

        logger.info("[LLM_STREAM] iterator_done -> sending stop signals to workers")
        for _ in range(TTS_WORKER_COUNT):
            await text_queue.put(STOP)
        logger.info(
            f"[LLM_STREAM] stop_signals_sent count={TTS_WORKER_COUNT} "
            f"text_queue_size={text_queue.qsize()}"
        )
        await text_queue.join()
        logger.info("[SYNC] text_queue joined")
        await audio_queue.join()
        logger.info("[SYNC] audio_queue joined")
        for task in tts_tasks:
            await task
        logger.info("[SYNC] all tts_workers joined")
        await sender_task
        logger.info("[SYNC] audio_sender joined")

        chat_history.append({"role": "user", "parts": [text_for_llm]})
        chat_history.append({"role": "model", "parts": [full_answer]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer})
        logger.info(
            f"[LLM_TTS_FLOW] complete answer_len={len(full_answer)} "
            f"total_llm_tts_ms={(time.perf_counter() - llm_tts_start)*1000.0:.1f}"
        )

    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}")
    finally:
        for task in tts_tasks:
            if not task.done():
                task.cancel()
        if not sender_task.done():
            sender_task.cancel()


# ---------------------------
# WebSocket エンドポイント
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected.")
    
    vad_iterator = VADIterator(
        vad_model, 
        threshold=0.95, 
        sampling_rate=16000, 
        min_silence_duration_ms=200, 
        speech_pad_ms=50
    )

    audio_buffer = [] 
    is_speaking = False
    interruption_triggered = False 
    
    WINDOW_SIZE_SAMPLES = 512 
    SAMPLE_RATE = 16000
    CHECK_SPEAKER_SAMPLES = 30000
    
    chat_history = []

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            audio_chunk_np = np.frombuffer(data_bytes, dtype=np.float32).copy()
            
            offset = 0
            while offset + WINDOW_SIZE_SAMPLES <= len(audio_chunk_np):
                window_np = audio_chunk_np[offset : offset + WINDOW_SIZE_SAMPLES]
                offset += WINDOW_SIZE_SAMPLES
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(DEVICE)

                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)
                
                if speech_dict:
                    if "start" in speech_dict:
                        logger.info("🗣️ Speech START")
                        is_speaking = True
                        interruption_triggered = False 
                        audio_buffer = [window_np]
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 Speech END")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            full_audio = np.concatenate(audio_buffer)
                            
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                logger.info("Noise detected")
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                await process_voice_pipeline(full_audio, websocket, chat_history)
                            audio_buffer = [] 
                else:
                    if is_speaking:
                        audio_buffer.append(window_np)
                        
                        current_len = sum(len(c) for c in audio_buffer)
                        if not interruption_triggered and not NEXT_AUDIO_IS_REGISTRATION and current_len > CHECK_SPEAKER_SAMPLES:
                            temp_audio = np.concatenate(audio_buffer)
                            temp_tensor = torch.from_numpy(temp_audio).float().unsqueeze(0)
                            
                            is_verified, spk_id = await asyncio.to_thread(speaker_guard.identify_speaker, temp_tensor)
                            
                            if is_verified:
                                logger.info(f"⚡ [Barge-in] {spk_id} の声を検知！停止指示。")
                                await websocket.send_json({"status": "interrupt", "message": "🛑 音声停止"})
                                interruption_triggered = True

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}", exc_info=True)
    finally:
        vad_iterator.reset_states()


# ---------------------------
# フロントエンド (Toast通知 & UI改善)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Team Chat AI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: grid; place-items: center; min-height: 90vh; background: #202c33; color: #e9edef; margin: 0; }
            #container { background: #111b21; padding: 0; border-radius: 0; text-align: center; width: 100%; max-width: 600px; height: 100vh; display: flex; flex-direction: column; box-shadow: 0 0 20px rgba(0,0,0,0.5); position: relative; overflow: hidden; }
            @media (min-width: 600px) {
                #container { height: 90vh; border-radius: 12px; }
            }
            
            header { background: #202c33; padding: 15px; border-bottom: 1px solid #374045; font-weight: bold; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center; z-index: 10; }
            
            #chat-box { 
                flex: 1; overflow-y: auto; padding: 20px; 
                background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
                background-repeat: repeat;
                background-size: 400px;
                background-color: #0b141a;
                position: relative;
            }

            .row { display: flex; width: 100%; margin-bottom: 8px; flex-direction: column; }
            .row.ai { align-items: flex-start; }
            .row.user { align-items: flex-end; }
            /* システムメッセージ用に行全体を中央揃えにする */
            .row.system { align-items: center; margin-bottom: 12px; }
            
            .speaker-name { font-size: 0.75rem; color: #8696a0; margin-bottom: 2px; margin-left: 5px; margin-right: 5px;}

            .bubble { 
                padding: 8px 12px; border-radius: 8px; max-width: 75%; 
                font-size: 0.95rem; line-height: 1.4; word-wrap: break-word;
                box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
            }
            .ai .bubble { background: #202c33; color: #e9edef; border-top-left-radius: 0; }
            
            /* ユーザー色分け */
            .user-type-0 .bubble { background: #005c4b; color: #e9edef; border-top-right-radius: 0; }
            .user-type-1 .bubble { background: #0078d4; color: #fff; border-top-right-radius: 0; }
            .user-type-2 .bubble { background: #6b63ff; color: #fff; border-top-right-radius: 0; }
            .user-type-unknown .bubble { background: #374045; color: #e9edef; border-top-right-radius: 0; }
            
            /* ★システム警告(無関係な内容)用スタイル - 視認性改善★ */
            .system-bubble {
                background: #4a3b00;         /* 暗めのオレンジ背景 */
                color: #ffecb3;              /* 明るいクリーム色の文字 */
                font-size: 0.85rem;
                padding: 6px 16px;
                border-radius: 16px;
                border: 1px solid #ffb300;   /* 明るいオレンジの枠線 */
                text-align: center;
                max-width: 90%;
                font-weight: 500;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            }

            /* ★未登録の声用 Toast通知スタイル★ */
            #toast-container {
                position: absolute;
                top: 70px; /* ヘッダーの下 */
                left: 50%;
                transform: translateX(-50%);
                z-index: 100;
                width: 90%;
                max-width: 400px;
                pointer-events: none; /* クリックを透過(ボタン以外) */
            }
            .toast {
                background: rgba(30, 30, 30, 0.95);
                color: #fff;
                padding: 12px 16px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                border-left: 4px solid #f44336; /* 赤いアクセント */
                margin-bottom: 10px;
                font-size: 0.9rem;
                display: flex;
                flex-direction: column;
                gap: 8px;
                opacity: 0;
                animation: slideDown 0.3s forwards, fadeOut 0.5s forwards 2.5s; /* 2.5秒後に消える */
                pointer-events: auto;
            }
            
            @keyframes slideDown { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
            @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }

            .toast-btn {
                align-self: flex-end;
                background: transparent;
                border: 1px solid #666;
                color: #ccc;
                font-size: 0.75rem;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
            }
            .toast-btn:hover { background: #333; color: #fff; }

            #controls { background: #202c33; padding: 15px; border-top: 1px solid #374045; }
            
            button { 
                padding: 10px 20px; border-radius: 24px; border: none; font-size: 1rem; cursor: pointer; margin: 0 5px; font-weight: bold; transition: opacity 0.2s;
            }
            button:active { opacity: 0.7; }
            #btn-start { background: #00a884; color: #fff; }
            #btn-stop { background: #ef5350; color: #fff; display: none; }
            #btn-register { background: #3b4a54; color: #fff; font-size: 0.8rem; padding: 8px 15px; }
            #status { margin-bottom: 10px; font-size: 0.9rem; color: #8696a0; min-height: 1.2em; }
        </style>
    </head>
    <body>
        <div id="container">
            <header>
                <span>Team Chat AI</span>
                <button id="btn-register">＋ メンバー追加</button>
            </header>
            
            <div id="toast-container"></div> <div id="chat-box"></div>
            
            <div id="controls">
                <div id="status">接続待機中...</div>
                <button id="btn-start">会話を始める</button>
                <button id="btn-stop">終了する</button>
            </div>
        </div>

        <script>
            let socket;
            let audioContext;
            let processor;
            let sourceInput;
            let isRecording = false;
            
            const btnStart = document.getElementById('btn-start');
            const btnStop = document.getElementById('btn-stop');
            const btnRegister = document.getElementById('btn-register');
            const statusDiv = document.getElementById('status');
            const chatBox = document.getElementById('chat-box');
            const toastContainer = document.getElementById('toast-container');

            let audioQueue = [];
            let audioMetaQueue = [];
            let pendingOrderedAudio = new Map();
            let sentenceDoneMap = new Map();
            let expectedSentenceId = 1;
            let expectedChunkId = 1;
            let isPlaying = false;
            let jitterPrimed = false;
            const JITTER_TARGET_MS = 320;   // 初回はこの分だけ貯めてから再生
            const JITTER_LOW_WATER_MS = 120; // 再生中にここを下回ったら再バッファ
            let currentSourceNode = null;
            let currentAiBubble = null;
            
            // ★「今後表示しない」設定
            let muteUnregisteredWarning = false;

            // --- Toast通知機能 ---
            function showToast(message) {
                if (muteUnregisteredWarning) return;

                const toast = document.createElement('div');
                toast.className = 'toast';
                
                const msgText = document.createElement('span');
                msgText.textContent = message;
                
                const muteBtn = document.createElement('button');
                muteBtn.className = 'toast-btn';
                muteBtn.textContent = "今後このメッセージを表示しない";
                muteBtn.onclick = () => {
                    muteUnregisteredWarning = true;
                    toast.style.display = 'none'; // 即座に消す
                };

                toast.appendChild(msgText);
                toast.appendChild(muteBtn);
                toastContainer.appendChild(toast);

                // アニメーション終了後にDOMから削除 (3s)
                setTimeout(() => {
                    if (toast.parentNode) toast.parentNode.removeChild(toast);
                }, 3000);
            }

            // --- チャットログ表示 ---
            function logChat(role, text, speakerId = null) {
                const row = document.createElement('div');
                row.className = `row ${role}`;
                
                const bubble = document.createElement('div');

                if (role === 'system') {
                    // システム(無関係)の場合は専用スタイル
                    bubble.className = 'system-bubble';
                    bubble.textContent = text;
                } else {
                    // 通常メッセージ
                    bubble.className = 'bubble';
                    bubble.textContent = text;
                    
                    if (role === 'user' && speakerId) {
                        const nameLabel = document.createElement('div');
                        nameLabel.className = 'speaker-name';
                        nameLabel.textContent = speakerId; 
                        row.insertBefore(nameLabel, row.firstChild); // 名前後入れ調整
                        
                        const idNum = speakerId.replace('User ', '');
                        if (!isNaN(idNum)) {
                            row.classList.add(`user-type-${idNum}`);
                        } else {
                            row.classList.add('user-type-unknown');
                        }
                    } else if (role === 'ai') {
                         const nameLabel = document.createElement('div');
                        nameLabel.className = 'speaker-name';
                        nameLabel.textContent = "AI Assistant";
                        row.insertBefore(nameLabel, row.firstChild);
                    }
                }
                
                row.appendChild(bubble);
                chatBox.appendChild(row);
                chatBox.scrollTop = chatBox.scrollHeight;
                return bubble;
            }

            btnRegister.onclick = async () => {
                try {
                    await fetch('/enable-registration', { method: 'POST' });
                    statusDiv.textContent = "🆕 新規メンバー登録モード";
                    statusDiv.style.color = "#00a884";
                    logChat('ai', "【システム】新しい方の声を登録します。マイクに向かって話しかけてください。");
                } catch(e) { console.error(e); }
            };

            function makeChunkKey(sentenceId, chunkId) {
                return `${sentenceId}:${chunkId}`;
            }

            function resetOrderedAudioState() {
                audioQueue = [];
                audioMetaQueue = [];
                pendingOrderedAudio.clear();
                sentenceDoneMap.clear();
                expectedSentenceId = 1;
                expectedChunkId = 1;
                nextStartTime = 0;
                isPlaying = false;
                jitterPrimed = false;
            }

            function getQueuedAudioMs() {
                let totalBytes = 0;
                for (const buf of audioQueue) {
                    totalBytes += buf.byteLength;
                }
                // PCM16 mono @16kHz: 2 bytes/sample
                const totalSamples = totalBytes / 2;
                return (totalSamples / 16000) * 1000;
            }

            function flushOrderedAudio() {
                while (true) {
                    const key = makeChunkKey(expectedSentenceId, expectedChunkId);
                    if (pendingOrderedAudio.has(key)) {
                        audioQueue.push(pendingOrderedAudio.get(key));
                        pendingOrderedAudio.delete(key);
                        expectedChunkId += 1;
                        if (!isPlaying) {
                            processAudioQueue();
                        }
                        continue;
                    }

                    const doneInfo = sentenceDoneMap.get(expectedSentenceId);
                    if (doneInfo && expectedChunkId > doneInfo.lastChunkId) {
                        expectedSentenceId += 1;
                        expectedChunkId = 1;
                        continue;
                    }
                    break;
                }
            }

            function queueOrderedChunk(meta, rawBytes) {
                const key = makeChunkKey(meta.sentence_id, meta.chunk_id);
                pendingOrderedAudio.set(key, rawBytes);
                flushOrderedAudio();
            }

            async function startRecording() {
                try {
                    statusDiv.textContent = "サーバー接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';

                    socket.onopen = async () => {
                        console.log("WS Connected");
                        statusDiv.textContent = "🎙️ 準備OK";
                        statusDiv.style.color = "#e9edef";
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        await initAudioStream();
                    };

                    socket.onmessage = async (event) => {
                        if (event.data instanceof ArrayBuffer) {
                            const meta = audioMetaQueue.shift();
                            if (meta) {
                                queueOrderedChunk(meta, event.data);
                            } else {
                                // Fallback for legacy binary packets without metadata.
                                audioQueue.push(event.data);
                                processAudioQueue();
                            }
                        } else {
                            const data = JSON.parse(event.data);
                            
                            if (data.status === 'processing') {
                                statusDiv.textContent = data.message;
                            }
                            if (data.status === 'interrupt') {
                                stopAudioPlayback();
                            }
                            if (data.status === 'audio_chunk_meta') {
                                audioMetaQueue.push(data);
                            }
                            if (data.status === 'audio_sentence_done') {
                                sentenceDoneMap.set(data.sentence_id, { lastChunkId: data.last_chunk_id });
                                flushOrderedAudio();
                            }
                            if (data.status === 'system_info') {
                                logChat('ai', data.message);
                            }

                            // ★ アラート分岐処理 ★
                            if (data.status === 'system_alert') {
                                if (data.alert_type === 'unregistered') {
                                    // 未登録 -> Toast表示
                                    showToast(data.message);
                                } else if (data.alert_type === 'irrelevant') {
                                    // 無関係 -> ログ表示(色調整済み)
                                    logChat('system', data.message);
                                }
                                statusDiv.textContent = "待機中...";
                            }

                            if (data.status === 'transcribed') {
                                logChat('user', data.question_text, data.speaker_id);
                            }

                            if (data.status === 'reply_chunk') {
                                if (!currentAiBubble) {
                                    currentAiBubble = logChat('ai', ''); 
                                }
                                currentAiBubble.textContent += data.text_chunk;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                            if (data.status === 'complete') {
                                if (!currentAiBubble && data.answer_text) {
                                    logChat('ai', data.answer_text);
                                }
                                currentAiBubble = null;
                                statusDiv.textContent = "🎙️ 準備OK";
                                // Keep ordered state clean between turns.
                                audioMetaQueue = [];
                                pendingOrderedAudio.clear();
                                sentenceDoneMap.clear();
                            }
                        }
                    };
                    socket.onclose = () => stopRecording();
                } catch (e) {
                    console.error(e);
                }
            }

            async function initAudioStream() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
                sourceInput = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(512, 1, 1);
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    socket.send(e.inputBuffer.getChannelData(0).buffer);
                };
                sourceInput.connect(processor);
                processor.connect(audioContext.destination);
                isRecording = true;
            }

            function stopRecording() {
                isRecording = false;
                if (sourceInput) sourceInput.disconnect();
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (socket) socket.close();
                btnStart.style.display = 'inline-block';
                btnStop.style.display = 'none';
                statusDiv.textContent = "停止中";
            }

            function stopAudioPlayback() {
                if (currentSourceNode) { try { currentSourceNode.stop(); } catch(e){} currentSourceNode = null; }
                resetOrderedAudioState();
            }

            // ★追加: 再生時間を管理する変数
            let nextStartTime = 0;

            async function processAudioQueue() {
                if (audioQueue.length === 0) {
                    isPlaying = false;
                    jitterPrimed = false;
                    return;
                }

                const queuedMs = getQueuedAudioMs();
                if (!jitterPrimed) {
                    if (queuedMs < JITTER_TARGET_MS) {
                        return;
                    }
                    jitterPrimed = true;
                } else if (queuedMs < JITTER_LOW_WATER_MS) {
                    // 低水位を下回ったら、少し貯まるまで再生を待つ
                    jitterPrimed = false;
                    return;
                }

                isPlaying = true;
                const rawBytes = audioQueue.shift();
                
                try {
                    if (audioContext.state === 'suspended') {
                        await audioContext.resume();
                    }

                    // --- ★ここが高速化のキモです ---
                    
                    // 1. 生のバイナリ(Int16)を読み込む
                    // サーバーから送られてきたのは 16bit整数 の配列です
                    const int16Data = new Int16Array(rawBytes);
                    
                    // 2. ブラウザ用に Float32 (-1.0 ~ 1.0) に変換する
                    // decodeAudioData を待つ必要がなく、計算だけで終わるため一瞬です
                    const float32Data = new Float32Array(int16Data.length);
                    for (let i = 0; i < int16Data.length; i++) {
                        // 32768で割って正規化
                        float32Data[i] = int16Data[i] / 32768.0;
                    }

                    // 3. 再生用バッファを作成 (モノラル, 長さ, 16000Hz)
                    // ※new_text_to_speech.py の target_sr と合わせる必要があります(今は16000推奨)
                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, 16000);
                    
                    // 4. データをバッファにコピー
                    audioBuffer.getChannelData(0).set(float32Data);

                    // 5. 隙間なく再生するスケジュール管理
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);

                    // 現在時刻と、予定時刻を比べて、遅れていれば現在時刻に合わせる
                    if (nextStartTime < audioContext.currentTime) {
                        nextStartTime = audioContext.currentTime;
                    }
                    
                    source.start(nextStartTime);
                    
                    // 次の音声の開始予定時間を更新（今の音声の長さ分だけ後ろにずらす）
                    nextStartTime += audioBuffer.duration;

                    // 再生が終わるのを待たずに、次のデータの準備にすぐ取り掛かる！
                    // (これで遅延がさらに減ります)
                    processAudioQueue();
                    
                } catch(e) { 
                    console.error("Raw再生エラー:", e);
                    isPlaying = false;
                }
            }

            btnStart.onclick = startRecording;
            btnStop.onclick = stopRecording;
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
