#今はこれ! 3月4日 

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
import wave

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
    import parallel_faster_text_to_speech as tts_module
    from parallel_faster_text_to_speech import (
        synthesize_speech,
        synthesize_speech_to_memory,
        synthesize_speech_to_memory_stream,
        synthesize_speech_to_memory_for_worker,
        synthesize_speech_to_memory_stream_for_worker,
    )
    from new_speaker_filter import SpeakerGuard
except ImportError as e:
    logger.error(f"[ERROR] 必要なモジュールが見つかりません: {e}")
    sys.exit(1)

# --- グローバル設定 ---
PROCESSING_DIR = "incoming_audio"
os.makedirs(PROCESSING_DIR, exist_ok=True)
TTS_DEBUG_WEB_DIR = os.path.join(PROCESSING_DIR, "tts_debug")
TTS_DEBUG_VIEWER_HTML = os.path.join(os.path.dirname(__file__), "tts_debug_browser.html")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using Device: {DEVICE}")
SYNC_ROOT_DIR = os.path.abspath(os.getenv("RUNPOD_SYNC_ROOT", "."))
SYNC_TOKEN = os.getenv("RUNPOD_SYNC_TOKEN", "").strip()
os.makedirs(SYNC_ROOT_DIR, exist_ok=True)
logger.info(f"[SYNC] root={SYNC_ROOT_DIR}")

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


def _verify_sync_token(token: str | None):
    if not SYNC_TOKEN:
        raise HTTPException(status_code=503, detail="Sync token is not configured on server.")
    if token != SYNC_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid sync token.")


def _resolve_sync_path(relative_path: str) -> str:
    rel = (relative_path or "").strip()
    if not rel:
        raise HTTPException(status_code=400, detail="relative_path is required.")
    rel = rel.lstrip("/").replace("\\", "/")
    full = os.path.abspath(os.path.join(SYNC_ROOT_DIR, rel))
    if not full.startswith(SYNC_ROOT_DIR + os.sep) and full != SYNC_ROOT_DIR:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.")
    return full


@app.post("/admin/upload-file")
async def upload_file(
    relative_path: str = Form(...),
    file: UploadFile = File(...),
    x_sync_token: str | None = Header(default=None),
):
    _verify_sync_token(x_sync_token)
    target_path = _resolve_sync_path(relative_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    data = await file.read()
    with open(target_path, "wb") as f:
        f.write(data)
    logger.info(f"[SYNC] uploaded: {target_path} ({len(data)} bytes)")
    return {"ok": True, "path": target_path, "bytes": len(data)}


@app.get("/api/tts-debug-files")
async def api_tts_debug_files():
    os.makedirs(TTS_DEBUG_WEB_DIR, exist_ok=True)
    rows = []
    for name in os.listdir(TTS_DEBUG_WEB_DIR):
        if not name.lower().endswith(".wav"):
            continue
        full = os.path.join(TTS_DEBUG_WEB_DIR, name)
        if not os.path.isfile(full):
            continue
        st = os.stat(full)
        rows.append(
            {
                "name": name,
                "size_bytes": int(st.st_size),
                "modified_ts": float(st.st_mtime),
                "url": f"/download/tts_debug/{name}",
            }
        )
    rows.sort(key=lambda x: x["modified_ts"], reverse=True)
    return JSONResponse({"files": rows, "dir": TTS_DEBUG_WEB_DIR})


@app.get("/tts-debug", response_class=HTMLResponse)
async def tts_debug_page():
    if not os.path.exists(TTS_DEBUG_VIEWER_HTML):
        return HTMLResponse(
            "<h3>tts_debug_browser.html が見つかりません。</h3>",
            status_code=500,
        )
    with open(TTS_DEBUG_VIEWER_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


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
    TTS_WORKER_COUNT = int(os.getenv("PERM_TTS_WORKER_COUNT", "2"))
    TTS_PREFETCH_AHEAD = 1
    TTS_MAX_CHUNKS_PER_SENTENCE = int(os.getenv("PERM_TTS_MAX_CHUNKS_PER_SENTENCE", "40"))
    STREAM_EMIT_EVERY_FRAMES = int(os.getenv("PERM_EMIT_EVERY_FRAMES", "4"))
    STREAM_DECODE_WINDOW_FRAMES = int(os.getenv("PERM_DECODE_WINDOW_FRAMES", "80"))
    DETAILED_TIMING = os.getenv("PERM_DETAILED_TIMING", "0") == "1"
    AUDIO_ENERGY_DIAG = os.getenv("PERM_AUDIO_ENERGY_DIAG", "0") == "1"
    TAIL_SILENCE_TRIM = os.getenv("PERM_TTS_TRIM_TAIL_SILENCE", "1") == "1"
    TAIL_SILENCE_DBFS = float(os.getenv("PERM_TTS_TAIL_SILENCE_DBFS", "-42.0"))
    # Backward compatible:
    # - old env: PERM_TTS_TAIL_SILENCE_MIN_CHUNKS (kept)
    # - new env: PERM_TTS_TAIL_SILENCE_MAX_TRIM_CHUNKS
    TAIL_SILENCE_MAX_TRIM_CHUNKS = int(
        os.getenv(
            "PERM_TTS_TAIL_SILENCE_MAX_TRIM_CHUNKS",
            os.getenv("PERM_TTS_TAIL_SILENCE_MIN_CHUNKS", "40"),
        )
    )
    TAIL_HOLD_CHUNKS = int(os.getenv("PERM_TTS_TAIL_HOLD_CHUNKS", "3"))
    TAIL_SILENCE_KEEP_CHUNKS = int(os.getenv("PERM_TTS_TAIL_SILENCE_KEEP_CHUNKS", "1"))
    HEAD_SILENCE_MAX_DROP_CHUNKS = int(os.getenv("PERM_TTS_HEAD_SILENCE_MAX_DROP_CHUNKS", "12"))
    HEAD_SILENCE_MAX_BUFFER_CHUNKS = int(os.getenv("PERM_TTS_HEAD_SILENCE_MAX_BUFFER_CHUNKS", "4"))
    # Optional: inject a small sentence-end gap when natural tail is too short.
    SENTENCE_GAP_MS = int(os.getenv("PERM_TTS_SENTENCE_GAP_MS", "0"))
    SENTENCE_GAP_DBFS = float(os.getenv("PERM_TTS_SENTENCE_GAP_DBFS", "-40.0"))
    SENTENCE_GAP_MAX_TAIL_LOW_CHUNKS = int(os.getenv("PERM_TTS_SENTENCE_GAP_MAX_TAIL_LOW_CHUNKS", "1"))
    SAVE_DEBUG_AUDIO = os.getenv("PERM_TTS_SAVE_DEBUG_AUDIO", "0") == "1"
    SAVE_DEBUG_AUDIO_DIR = os.getenv("PERM_TTS_SAVE_DEBUG_AUDIO_DIR", os.path.join(PROCESSING_DIR, "tts_debug"))
    turn_id = int(time.time() * 1000)
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
        f"tts_workers={TTS_WORKER_COUNT} prefetch_ahead={TTS_PREFETCH_AHEAD} "
        f"max_chunks_per_sentence={TTS_MAX_CHUNKS_PER_SENTENCE} "
        f"audio_energy_diag={AUDIO_ENERGY_DIAG} "
        f"trim_tail_silence={TAIL_SILENCE_TRIM} "
        f"tail_silence_dbfs={TAIL_SILENCE_DBFS} "
        f"tail_silence_max_trim_chunks={TAIL_SILENCE_MAX_TRIM_CHUNKS} "
        f"tail_hold_chunks={TAIL_HOLD_CHUNKS} "
        f"tail_silence_keep_chunks={TAIL_SILENCE_KEEP_CHUNKS} "
        f"head_silence_max_drop_chunks={HEAD_SILENCE_MAX_DROP_CHUNKS} "
        f"head_silence_max_buffer_chunks={HEAD_SILENCE_MAX_BUFFER_CHUNKS} "
        f"sentence_gap_ms={SENTENCE_GAP_MS} "
        f"sentence_gap_dbfs={SENTENCE_GAP_DBFS} "
        f"sentence_gap_max_tail_low_chunks={SENTENCE_GAP_MAX_TAIL_LOW_CHUNKS} "
        f"save_debug_audio={SAVE_DEBUG_AUDIO} "
        f"save_debug_audio_dir={SAVE_DEBUG_AUDIO_DIR}"
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

    def _save_debug_wav(path: str, pcm_bytes: bytes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)

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
                phrase_preview = phrase[:80].replace("\n", "\\n")
                tts_snapshot = None
                if hasattr(tts_module, "get_tts_debug_snapshot"):
                    try:
                        tts_snapshot = tts_module.get_tts_debug_snapshot(worker_id)
                    except Exception:
                        tts_snapshot = {"snapshot_error": True}
                logger.info(
                    f"[TTS_DIAG] worker={worker_id} sentence={idx} "
                    f"phrase={phrase_preview!r} phrase_repr={phrase!r} model={tts_snapshot}"
                )
                total_len = 0
                tts_chunk_count = 0
                first_chunk_ready_ms = None
                termination_reason = "stream_stop"
                tail_low_energy_chunks = 0
                low_energy_dbfs = TAIL_SILENCE_DBFS
                tail_silence_buffer = []
                head_silence_buffer = []
                emitted_non_silence = False
                head_dropped_chunks = 0
                sentence_pcm = bytearray()
                raw_tail_low_energy_chunks = 0

                def _calc_chunk_dbfs(pcm_chunk: bytes):
                    if not pcm_chunk:
                        return None
                    arr = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    if arr.size == 0:
                        return None
                    rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
                    return 20.0 * np.log10(max(rms, 1e-8))

                async def _emit_pcm_chunk(pcm_chunk: bytes, chunk_gen_ms_value: float):
                    nonlocal tts_chunk_count, total_len, first_chunk_ready_ms, tail_low_energy_chunks
                    if not pcm_chunk:
                        return None
                    tts_chunk_count += 1
                    chunk_dbfs_local = None
                    arr = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    if arr.size > 0:
                        rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
                        chunk_dbfs_local = 20.0 * np.log10(max(rms, 1e-8))
                        if chunk_dbfs_local < low_energy_dbfs:
                            tail_low_energy_chunks += 1
                        else:
                            tail_low_energy_chunks = 0
                    if first_chunk_ready_ms is None:
                        first_chunk_ready_ms = (time.perf_counter() - sentence_start) * 1000.0
                    total_len += len(pcm_chunk)
                    sentence_pcm.extend(pcm_chunk)
                    created_at = time.perf_counter()
                    await audio_queue.put(
                        {
                            "type": "chunk",
                            "sentence_idx": idx,
                            "tts_chunk_idx": tts_chunk_count,
                            "worker_id": worker_id,
                            "audio_bytes": pcm_chunk,
                            "total_bytes": 0,
                            "chunk_gen_ms": chunk_gen_ms_value,
                            "created_at": created_at,
                        }
                    )
                    logger.info(
                        f"[TTS_TIMING] worker={worker_id} sentence={idx} tts_chunk={tts_chunk_count} "
                        f"chunk_bytes={len(pcm_chunk)} chunk_gen_ms={chunk_gen_ms_value:.1f}"
                    )
                    if AUDIO_ENERGY_DIAG and chunk_dbfs_local is not None:
                        logger.info(
                            f"[TTS_AUDIO] worker={worker_id} sentence={idx} tts_chunk={tts_chunk_count} "
                            f"chunk_dbfs={chunk_dbfs_local:.1f} tail_low_energy_chunks={tail_low_energy_chunks}"
                        )
                    return chunk_dbfs_local

                try:
                    stream_gen = synthesize_speech_to_memory_stream_for_worker(phrase, worker_id)
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
                            termination_reason = "stream_eos"
                            break
                        raw_dbfs_now = _calc_chunk_dbfs(pcm_chunk)
                        if raw_dbfs_now is not None and raw_dbfs_now < SENTENCE_GAP_DBFS:
                            raw_tail_low_energy_chunks += 1
                        else:
                            raw_tail_low_energy_chunks = 0
                        if TAIL_SILENCE_TRIM:
                            chunk_dbfs_now = _calc_chunk_dbfs(pcm_chunk)
                            if chunk_dbfs_now is not None and chunk_dbfs_now < low_energy_dbfs:
                                if not emitted_non_silence:
                                    # 先頭の低エネルギー区間: まずは削除、削除上限を超えたら最小限だけ送る。
                                    if head_dropped_chunks < HEAD_SILENCE_MAX_DROP_CHUNKS:
                                        head_dropped_chunks += 1
                                        continue
                                    head_silence_buffer.append((pcm_chunk, chunk_gen_ms))
                                    if len(head_silence_buffer) > max(1, HEAD_SILENCE_MAX_BUFFER_CHUNKS):
                                        emit_pcm, emit_ms = head_silence_buffer.pop(0)
                                        await _emit_pcm_chunk(emit_pcm, emit_ms)
                                    continue
                                # 非無音出現後の低エネルギー区間は末尾候補として保持
                                tail_silence_buffer.append((pcm_chunk, chunk_gen_ms))
                                continue
                            if not emitted_non_silence:
                                emitted_non_silence = True
                                if head_silence_buffer:
                                    for emit_pcm, emit_ms in head_silence_buffer:
                                        await _emit_pcm_chunk(emit_pcm, emit_ms)
                                    head_silence_buffer.clear()
                            # 非無音が来たので、保持していた末尾候補は実際には「末尾」でない。
                            if tail_silence_buffer:
                                for emit_pcm, emit_ms in tail_silence_buffer:
                                    await _emit_pcm_chunk(emit_pcm, emit_ms)
                                tail_silence_buffer.clear()
                            await _emit_pcm_chunk(pcm_chunk, chunk_gen_ms)
                        else:
                            await _emit_pcm_chunk(pcm_chunk, chunk_gen_ms)
                        if tts_chunk_count >= TTS_MAX_CHUNKS_PER_SENTENCE:
                            logger.warning(
                                f"[TTS_TIMING] worker={worker_id} sentence={idx} "
                                f"hit_chunk_cap={TTS_MAX_CHUNKS_PER_SENTENCE} -> truncating sentence"
                            )
                            termination_reason = "chunk_cap"
                            break
                        if DETAILED_TIMING:
                            logger.info(
                                f"[TTS_DETAILED] worker={worker_id} sentence={idx} chunk={tts_chunk_count} "
                                f"to_thread_overhead_ms={to_thread_overhead_ms:.3f} "
                                f"approx_model_ms={approx_model_ms:.1f}"
                            )
                    if TAIL_SILENCE_TRIM and tail_silence_buffer:
                        # EOS時点で保持中なのは「文末連続低エネルギー区間」。
                        # 上限まで削り、少しだけ残したい場合は KEEP_CHUNKS で制御。
                        drop_cap = max(0, TAIL_SILENCE_MAX_TRIM_CHUNKS)
                        keep_chunks = max(0, TAIL_SILENCE_KEEP_CHUNKS)
                        drop_count = min(max(0, len(tail_silence_buffer) - keep_chunks), drop_cap)
                        if drop_count > 0:
                            logger.info(
                                f"[TTS_AUDIO] worker={worker_id} sentence={idx} "
                                f"tail_trimmed_chunks={drop_count} threshold_dbfs={low_energy_dbfs:.1f} "
                                f"buffered_tail_chunks={len(tail_silence_buffer)} keep_chunks={keep_chunks}"
                            )
                        for emit_pcm, emit_ms in tail_silence_buffer[: len(tail_silence_buffer) - drop_count]:
                            await _emit_pcm_chunk(emit_pcm, emit_ms)
                        tail_silence_buffer.clear()
                    if TAIL_SILENCE_TRIM and (head_dropped_chunks > 0 or head_silence_buffer):
                        logger.info(
                            f"[TTS_AUDIO] worker={worker_id} sentence={idx} "
                            f"head_dropped_chunks={head_dropped_chunks} "
                            f"head_buffered_chunks={len(head_silence_buffer)} threshold_dbfs={low_energy_dbfs:.1f}"
                        )
                    if head_silence_buffer:
                        for emit_pcm, emit_ms in head_silence_buffer:
                            await _emit_pcm_chunk(emit_pcm, emit_ms)
                        head_silence_buffer.clear()
                except Exception as e:
                    termination_reason = f"stream_exception:{type(e).__name__}"
                    logger.error(
                        f"[TTS_TIMING] worker={worker_id} sentence={idx} stream_tts_failed: {e}",
                        exc_info=True,
                    )

                # ストリーミングで1チャンクも取れなかった場合は非ストリーミングへフォールバック
                if tts_chunk_count == 0:
                    logger.warning(
                        f"[TTS_TIMING] worker={worker_id} sentence={idx} no_stream_chunk -> fallback_non_streaming"
                    )
                    termination_reason = "fallback_non_streaming"
                    fallback_start = time.perf_counter()
                    pcm_all = await asyncio.to_thread(
                        synthesize_speech_to_memory_for_worker,
                        phrase,
                        worker_id,
                    )
                    fallback_ms = (time.perf_counter() - fallback_start) * 1000.0
                    if pcm_all:
                        tts_chunk_count = 1
                        total_len = len(pcm_all)
                        sentence_pcm = bytearray(pcm_all)
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
                        termination_reason = "fallback_empty"
                        logger.warning(
                            f"[TTS_TIMING] worker={worker_id} sentence={idx} fallback returned empty pcm"
                        )

                total_tts_ms = (time.perf_counter() - sentence_start) * 1000.0
                if SENTENCE_GAP_MS > 0 and raw_tail_low_energy_chunks <= SENTENCE_GAP_MAX_TAIL_LOW_CHUNKS:
                    gap_samples = int((SAMPLE_RATE * SENTENCE_GAP_MS) / 1000)
                    if gap_samples > 0:
                        gap_pcm = (np.zeros(gap_samples, dtype=np.int16)).tobytes()
                        if gap_pcm:
                            tts_chunk_count += 1
                            total_len += len(gap_pcm)
                            sentence_pcm.extend(gap_pcm)
                            await audio_queue.put(
                                {
                                    "type": "chunk",
                                    "sentence_idx": idx,
                                    "tts_chunk_idx": tts_chunk_count,
                                    "worker_id": worker_id,
                                    "audio_bytes": gap_pcm,
                                    "total_bytes": 0,
                                    "chunk_gen_ms": 0.0,
                                    "created_at": time.perf_counter(),
                                }
                            )
                            logger.info(
                                f"[TTS_GAP] worker={worker_id} sentence={idx} "
                                f"added_silence_ms={SENTENCE_GAP_MS} gap_bytes={len(gap_pcm)} "
                                f"raw_tail_low_energy_chunks={raw_tail_low_energy_chunks} "
                                f"threshold_dbfs={SENTENCE_GAP_DBFS:.1f}"
                            )
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
                    f"tts_chunk_count={tts_chunk_count} total_bytes={total_len} "
                    f"total_tts_ms={total_tts_ms:.1f} termination={termination_reason}"
                )
                if SAVE_DEBUG_AUDIO and total_len > 0:
                    debug_wav_path = os.path.join(
                        SAVE_DEBUG_AUDIO_DIR,
                        f"turn_{turn_id}_sentence_{idx:02d}_worker_{worker_id}_bytes_{total_len}.wav",
                    )
                    try:
                        await asyncio.to_thread(_save_debug_wav, debug_wav_path, bytes(sentence_pcm))
                        logger.info(
                            f"[TTS_DEBUG_AUDIO] worker={worker_id} sentence={idx} "
                            f"saved_wav={debug_wav_path} bytes={total_len}"
                        )
                    except Exception as save_e:
                        logger.error(
                            f"[TTS_DEBUG_AUDIO] worker={worker_id} sentence={idx} save_failed: {save_e}",
                            exc_info=True,
                        )
                if AUDIO_ENERGY_DIAG:
                    logger.info(
                        f"[TTS_AUDIO] worker={worker_id} sentence={idx} stage=summary "
                        f"tail_low_energy_chunks={tail_low_energy_chunks} threshold_dbfs={low_energy_dbfs:.1f}"
                    )
            finally:
                text_queue.task_done()

    async def audio_sender_worker():
        nonlocal first_audio_sent_at
        sent_arrival_seq = 0
        global_chunk_id = 0
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
                    global_chunk_id += 1
                    await websocket.send_json(
                        {
                            "status": "audio_chunk_meta",
                            "sentence_id": idx,
                            "chunk_id": tts_chunk_idx,
                            "global_chunk_id": global_chunk_id,
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
            let processLoopActive = false;
            let jitterPrimed = false;
            const JITTER_TARGET_MS = 320;   // ターン先頭のみ、この分だけ貯めてから再生
            let currentSourceNode = null;
            let currentAiBubble = null;
            const AUDIO_DIAG = true;
            let lastScheduledSentenceId = null;
            
            // ★「今後表示しない」設定
            let muteUnregisteredWarning = false;

            function dlog(...args) {
                if (!AUDIO_DIAG) return;
                console.log("[AUDIO_DIAG]", ...args);
            }

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

            function resetOrderedAudioState() {
                audioQueue = [];
                audioMetaQueue = [];
                pendingOrderedAudio.clear();
                sentenceDoneMap.clear();
                expectedSentenceId = 1;
                expectedChunkId = 1;
                nextStartTime = 0;
                isPlaying = false;
                processLoopActive = false;
                // ターン切替時のみジッタ初期化
                jitterPrimed = false;
                lastScheduledSentenceId = null;
            }

            function makeChunkKey(sentenceId, chunkId) {
                return `${sentenceId}:${chunkId}`;
            }

            function getQueuedAudioMs() {
                let totalBytes = 0;
                for (const pkt of audioQueue) {
                    if (pkt && pkt.rawBytes) {
                        totalBytes += pkt.rawBytes.byteLength;
                    }
                }
                // PCM16 mono @16kHz: 2 bytes/sample
                const totalSamples = totalBytes / 2;
                return (totalSamples / 16000) * 1000;
            }

            function getScheduledAheadMs() {
                if (!audioContext) return 0;
                return Math.max(0, (nextStartTime - audioContext.currentTime) * 1000);
            }

            function getBufferedAudioMs() {
                return getQueuedAudioMs() + getScheduledAheadMs();
            }

            function flushOrderedAudio() {
                while (true) {
                    const key = makeChunkKey(expectedSentenceId, expectedChunkId);
                    if (pendingOrderedAudio.has(key)) {
                        audioQueue.push(pendingOrderedAudio.get(key));
                        pendingOrderedAudio.delete(key);
                        dlog(
                            "flushOrderedAudio push",
                            "sentence=", expectedSentenceId,
                            "chunk=", expectedChunkId,
                            "audioQueue=", audioQueue.length,
                            "pending=", pendingOrderedAudio.size
                        );
                        expectedChunkId += 1;
                        processAudioQueue();
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
                pendingOrderedAudio.set(key, { meta, rawBytes, enqueuedAt: performance.now() });
                dlog(
                    "queueOrderedChunk",
                    "sentence=", meta.sentence_id,
                    "chunk=", meta.chunk_id,
                    "global=", meta.global_chunk_id,
                    "metaQ=", audioMetaQueue.length,
                    "pending=", pendingOrderedAudio.size
                );
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
                                dlog("binary_without_meta", "bytes=", event.data.byteLength);
                                audioQueue.push({ meta: null, rawBytes: event.data, enqueuedAt: performance.now() });
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
                                dlog(
                                    "meta_received",
                                    "sentence=", data.sentence_id,
                                    "chunk=", data.chunk_id,
                                    "global=", data.global_chunk_id,
                                    "metaQ=", audioMetaQueue.length
                                );
                            }
                            if (data.status === 'audio_sentence_done') {
                                sentenceDoneMap.set(data.sentence_id, { lastChunkId: data.last_chunk_id });
                                flushOrderedAudio();
                                dlog(
                                    "sentence_done",
                                    "sentence=", data.sentence_id,
                                    "last_chunk=", data.last_chunk_id,
                                    "expected_sentence=", expectedSentenceId,
                                    "expected_chunk=", expectedChunkId
                                );
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
                                // Fully reset ordered playback state between turns.
                                resetOrderedAudioState();
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
                if (processLoopActive) return;
                processLoopActive = true;
                try {
                    while (true) {
                if (audioQueue.length === 0) {
                    if (getScheduledAheadMs() > 0) {
                        return;
                    }
                    isPlaying = false;
                    // ここでfalseに戻すと文境界ごとに再バッファ待ちが入るため維持する
                    return;
                }

                const queuedMs = getBufferedAudioMs();
                if (!jitterPrimed) {
                    if (queuedMs < JITTER_TARGET_MS) {
                        dlog("jitter_wait", "bufferedMs=", queuedMs.toFixed(1), "target=", JITTER_TARGET_MS);
                        return;
                    }
                    jitterPrimed = true;
                }

                isPlaying = true;
                const packet = audioQueue.shift();
                const rawBytes = packet.rawBytes;
                
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
                    const underrunMs = Math.max(0, (audioContext.currentTime - nextStartTime) * 1000);
                    if (nextStartTime < audioContext.currentTime) {
                        nextStartTime = audioContext.currentTime;
                    }
                    const sentenceId = packet.meta?.sentence_id ?? null;
                    const chunkId = packet.meta?.chunk_id ?? null;
                    const globalChunkId = packet.meta?.global_chunk_id ?? null;
                    if (lastScheduledSentenceId !== null && sentenceId !== null && sentenceId !== lastScheduledSentenceId) {
                        dlog(
                            "sentence_switch",
                            "from=", lastScheduledSentenceId,
                            "to=", sentenceId,
                            "underrunMs=", underrunMs.toFixed(1)
                        );
                    }
                    lastScheduledSentenceId = sentenceId;
                    dlog(
                        "schedule_chunk",
                        "sentence=", sentenceId,
                        "chunk=", chunkId,
                        "global=", globalChunkId,
                        "bytes=", rawBytes.byteLength,
                        "queuedForMs=", (performance.now() - packet.enqueuedAt).toFixed(1),
                        "underrunMs=", underrunMs.toFixed(1),
                        "audioQueue=", audioQueue.length
                    );
                    
                    source.start(nextStartTime);
                    
                    // 次の音声の開始予定時間を更新（今の音声の長さ分だけ後ろにずらす）
                    nextStartTime += audioBuffer.duration;

                } catch(e) { 
                    console.error("Raw再生エラー:", e);
                    isPlaying = false;
                    return;
                }
            }
                } finally {
                    processLoopActive = false;
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
