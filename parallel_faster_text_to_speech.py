import io
import os
import sys
import threading
import time
import traceback
import hashlib

import numpy as np
import scipy.signal
import soundfile as sf
import torch

# Allow local repo usage without pip install.
_THIS_DIR = os.path.dirname(__file__)
_FASTER_REPO = os.path.abspath(os.path.join(_THIS_DIR, "..", "faster-qwen3-tts"))
if os.path.isdir(_FASTER_REPO) and _FASTER_REPO not in sys.path:
    sys.path.insert(0, _FASTER_REPO)

from faster_qwen3_tts import FasterQwen3TTS


GLOBAL_TTS_MODEL = None
GLOBAL_TTS_MODELS = []
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARALLEL_TTS_MODEL_COUNT = int(os.getenv("PARALLEL_TTS_MODEL_COUNT", "2" if "cuda" in GLOBAL_DEVICE else "1"))
GLOBAL_WARMUP_LOCK = threading.Lock()
GLOBAL_MODEL_STATE_LOCK = threading.Lock()
GLOBAL_MODEL_WARMUP_LOCKS = {}
GLOBAL_MODEL_WARMED = {}

QWEN3_MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
QWEN3_REF_AUDIO = os.getenv("QWEN3_REF_AUDIO", "")
QWEN3_REF_TEXT = os.getenv("QWEN3_REF_TEXT", "")
QWEN3_LANGUAGE = os.getenv("QWEN3_LANGUAGE", "Japanese")
QWEN3_DTYPE = os.getenv("QWEN3_DTYPE", "bfloat16")
QWEN3_XVECTOR_ONLY = os.getenv("QWEN3_XVECTOR_ONLY", "1") == "1"
QWEN3_STAGE_TIMING = os.getenv("QWEN3_STAGE_TIMING", "0") == "1"
QWEN3_DIAG = os.getenv("QWEN3_DIAG", "0") == "1"
QWEN3_DYNAMIC_MAX_NEW_TOKENS = os.getenv("QWEN3_DYNAMIC_MAX_NEW_TOKENS", "1") == "1"
QWEN3_MIN_NEW_TOKENS = int(os.getenv("QWEN3_MIN_NEW_TOKENS", "96"))
QWEN3_MAX_NEW_TOKENS_CAP = int(os.getenv("QWEN3_MAX_NEW_TOKENS_CAP", "320"))
QWEN3_NEW_TOKENS_PER_CHAR = float(os.getenv("QWEN3_NEW_TOKENS_PER_CHAR", "6.0"))
QWEN3_NEW_TOKENS_BIAS = int(os.getenv("QWEN3_NEW_TOKENS_BIAS", "48"))

DEFAULT_PARAMS = {
    "instruct": "人間らしく、感情豊かに、自然な息遣いで話してください。文末をはっきりと発音すること！",
    "do_sample": False,
    "temperature": 0.95,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "target_sr": 16000,
    "seed": 42,
}

# Keep keys compatible with existing caller (faster_main.py).
DEFAULT_STREAM_PARAMS = {
    "emit_every_frames": 8,  # mapped to faster-qwen3-tts chunk_size
    "decode_window_frames": 80,
    "overlap_samples": 512,
    "max_frames": 10000,
    "first_chunk_emit_every": 0,
    "first_chunk_decode_window": 48,
    "first_chunk_frames": 48,
    "repetition_penalty": 1.0,
    "repetition_penalty_window": 100,
    "use_optimized_decode": True,
}


def _resolve_dtype(name: str):
    key = str(name).lower()
    if key == "bfloat16":
        return torch.bfloat16
    if key == "float16":
        return torch.float16
    return torch.float32


def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


def _resample_if_needed(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    num_samples = int(len(audio) * float(tgt_sr) / float(src_sr))
    if num_samples <= 0:
        return audio
    return scipy.signal.resample(audio, num_samples).astype(np.float32)


def _estimate_max_new_tokens(text: str) -> int:
    if not QWEN3_DYNAMIC_MAX_NEW_TOKENS:
        return int(DEFAULT_PARAMS["max_new_tokens"])
    text_len = max(1, len(text.strip()))
    est = int(QWEN3_NEW_TOKENS_BIAS + (text_len * QWEN3_NEW_TOKENS_PER_CHAR))
    est = max(QWEN3_MIN_NEW_TOKENS, est)
    est = min(QWEN3_MAX_NEW_TOKENS_CAP, est)
    return est


def _pick_model(worker_id: int | None = None):
    if not GLOBAL_TTS_MODELS:
        return None
    if worker_id is None:
        return GLOBAL_TTS_MODELS[0]
    idx = (int(worker_id) - 1) % len(GLOBAL_TTS_MODELS)
    return GLOBAL_TTS_MODELS[idx]


def get_tts_debug_snapshot(worker_id: int | None = None) -> dict:
    tts_model = _pick_model(worker_id)
    if tts_model is None:
        return {
            "worker_id": worker_id,
            "model_exists": False,
        }
    predictor = getattr(tts_model, "predictor_graph", None)
    talker = getattr(tts_model, "talker_graph", None)
    return {
        "worker_id": worker_id,
        "model_exists": True,
        "model_id": id(tts_model),
        "model_warmed": _is_model_warmed(tts_model),
        "predictor_graph_id": id(predictor) if predictor is not None else None,
        "talker_graph_id": id(talker) if talker is not None else None,
        "predictor_do_sample": getattr(predictor, "do_sample", None) if predictor is not None else None,
        "max_new_tokens": DEFAULT_PARAMS["max_new_tokens"],
        "emit_every_frames": DEFAULT_STREAM_PARAMS.get("emit_every_frames"),
    }


def _get_model_warmup_lock(tts_model):
    mid = id(tts_model)
    with GLOBAL_MODEL_STATE_LOCK:
        if mid not in GLOBAL_MODEL_WARMUP_LOCKS:
            GLOBAL_MODEL_WARMUP_LOCKS[mid] = threading.Lock()
        return GLOBAL_MODEL_WARMUP_LOCKS[mid]


def _is_model_warmed(tts_model) -> bool:
    if tts_model is None:
        return False
    if bool(getattr(tts_model, "_warmed_up", False)):
        return True
    with GLOBAL_MODEL_STATE_LOCK:
        return bool(GLOBAL_MODEL_WARMED.get(id(tts_model), False))


def _mark_model_warmed(tts_model):
    if tts_model is None:
        return
    with GLOBAL_MODEL_STATE_LOCK:
        GLOBAL_MODEL_WARMED[id(tts_model)] = True


def _ensure_model_warm(tts_model, model_idx: int | None = None):
    if tts_model is None:
        raise RuntimeError("Faster Qwen3-TTS model is not initialized.")
    if _is_model_warmed(tts_model):
        return

    model_lock = _get_model_warmup_lock(tts_model)
    with model_lock:
        if _is_model_warmed(tts_model):
            return
        # CUDA graph capture is not safe to run concurrently.
        with GLOBAL_WARMUP_LOCK:
            if _is_model_warmed(tts_model):
                return
            tag = f"model={model_idx}" if model_idx is not None else "model=unknown"
            print(f"[INFO] Faster Qwen3-TTS: warm-up start ({tag})")
            _ = tts_model.generate_voice_clone(
                text="あ",
                language=QWEN3_LANGUAGE,
                ref_audio=QWEN3_REF_AUDIO,
                ref_text=QWEN3_REF_TEXT,
                do_sample=False,
                max_new_tokens=96,
                xvec_only=QWEN3_XVECTOR_ONLY,
            )
            _mark_model_warmed(tts_model)
            print(f"[INFO] Faster Qwen3-TTS: warm-up done ({tag})")


def _generate_wav_with_model(tts_model, text_to_speak: str, prompt_text: str = None):
    if tts_model is None:
        raise RuntimeError("Faster Qwen3-TTS model is not initialized.")
    if not QWEN3_REF_AUDIO:
        raise RuntimeError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if prompt_text:
        print("[WARN] prompt_text/instruct is ignored in voice-clone mode.")

    _ensure_model_warm(tts_model)
    max_new_tokens = _estimate_max_new_tokens(text_to_speak)
    if QWEN3_DIAG:
        print(
            f"[TTS_DIAG] nonstream_decode text_len={len(text_to_speak)} "
            f"max_new_tokens={max_new_tokens}"
        )

    wavs, sr = tts_model.generate_voice_clone(
        text=text_to_speak,
        language=QWEN3_LANGUAGE,
        ref_audio=QWEN3_REF_AUDIO,
        ref_text=QWEN3_REF_TEXT,
        do_sample=DEFAULT_PARAMS["do_sample"],
        temperature=DEFAULT_PARAMS["temperature"],
        top_p=DEFAULT_PARAMS["top_p"],
        top_k=DEFAULT_PARAMS["top_k"],
        repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
        max_new_tokens=max_new_tokens,
        xvec_only=QWEN3_XVECTOR_ONLY,
    )
    if len(wavs) == 0:
        raise RuntimeError("No waveform was generated.")
    return np.asarray(wavs[0], dtype=np.float32), int(sr)


def _generate_wav(text_to_speak: str, prompt_text: str = None):
    return _generate_wav_with_model(_pick_model(None), text_to_speak, prompt_text)


def _generate_wav_with_timing(text_to_speak: str, prompt_text: str = None):
    timings = {"generate_ms": None, "decode_ms": None, "total_ms": None}
    t0 = time.perf_counter()
    wav, sr = _generate_wav(text_to_speak, prompt_text=prompt_text)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
    return wav, sr, timings


def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    try:
        wav, sr = _generate_wav(text_to_speak, prompt_text=prompt_text)
        os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
        sf.write(output_wav_path, wav, sr)
        print(f"[SUCCESS] Saved to {output_wav_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        traceback.print_exc()
        return False


def synthesize_speech_to_memory(text_to_speak: str) -> bytes:
    try:
        wav, sr = _generate_wav(text_to_speak, prompt_text=None)
        wav = _resample_if_needed(wav, sr, DEFAULT_PARAMS["target_sr"])
        return _to_pcm16_bytes(wav)
    except Exception as e:
        print(f"[ERROR] Memory synthesis failed: {e}")
        traceback.print_exc()
        return None


def synthesize_speech_to_memory_stream(text_to_speak: str, prompt_text: str = None):
    tts_model = _pick_model(None)
    if tts_model is None:
        raise RuntimeError("Faster Qwen3-TTS model is not initialized.")
    if not QWEN3_REF_AUDIO:
        raise RuntimeError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if prompt_text:
        print("[WARN] prompt_text/instruct is ignored in streaming mode.")

    _ensure_model_warm(tts_model)

    chunk_size = max(1, int(DEFAULT_STREAM_PARAMS.get("emit_every_frames", 8)))

    chunk_iter = tts_model.generate_voice_clone_streaming(
        text=text_to_speak,
        language=QWEN3_LANGUAGE,
        ref_audio=QWEN3_REF_AUDIO,
        ref_text=QWEN3_REF_TEXT,
        do_sample=DEFAULT_PARAMS["do_sample"],
        temperature=DEFAULT_PARAMS["temperature"],
        top_p=DEFAULT_PARAMS["top_p"],
        top_k=DEFAULT_PARAMS["top_k"],
        repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
        max_new_tokens=DEFAULT_PARAMS["max_new_tokens"],
        chunk_size=chunk_size,
        xvec_only=QWEN3_XVECTOR_ONLY,
    )

    chunk_idx = 0
    for wav_chunk, sr, _timing in chunk_iter:
        chunk_idx += 1
        wav_chunk = np.asarray(wav_chunk, dtype=np.float32)
        if wav_chunk.size == 0:
            if QWEN3_STAGE_TIMING:
                print(f"[TTS_STAGE] chunk={chunk_idx} empty=1")
            continue

        wav_chunk = _resample_if_needed(wav_chunk, int(sr), DEFAULT_PARAMS["target_sr"])
        pcm = _to_pcm16_bytes(wav_chunk)

        if QWEN3_STAGE_TIMING:
            print(
                f"[TTS_STAGE] chunk={chunk_idx} sr_in={int(sr)} "
                f"sr_out={DEFAULT_PARAMS['target_sr']} pcm_bytes={len(pcm) if pcm else 0}"
            )

        if pcm:
            yield pcm


def synthesize_speech_to_memory_for_worker(text_to_speak: str, worker_id: int) -> bytes:
    try:
        tts_model = _pick_model(worker_id)
        if QWEN3_DIAG:
            snap = get_tts_debug_snapshot(worker_id)
            print(f"[TTS_DIAG] nonstream_start worker={worker_id} text_len={len(text_to_speak)} snap={snap}")
        wav, sr = _generate_wav_with_model(tts_model, text_to_speak, prompt_text=None)
        wav = _resample_if_needed(wav, sr, DEFAULT_PARAMS["target_sr"])
        return _to_pcm16_bytes(wav)
    except Exception as e:
        print(f"[ERROR] Memory synthesis failed (worker={worker_id}): {e}")
        traceback.print_exc()
        return None


def synthesize_speech_to_memory_stream_for_worker(text_to_speak: str, worker_id: int, prompt_text: str = None):
    tts_model = _pick_model(worker_id)
    if tts_model is None:
        raise RuntimeError(f"Faster Qwen3-TTS model is not initialized (worker={worker_id}).")
    if not QWEN3_REF_AUDIO:
        raise RuntimeError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if prompt_text:
        print("[WARN] prompt_text/instruct is ignored in streaming mode.")

    text_hash = hashlib.sha1(text_to_speak.encode("utf-8", errors="ignore")).hexdigest()[:10]
    stream_id = f"w{worker_id}-{int(time.time() * 1000)}-{text_hash}"
    if QWEN3_DIAG:
        snap = get_tts_debug_snapshot(worker_id)
        preview = text_to_speak[:120].replace("\n", "\\n")
        print(
            f"[TTS_DIAG] stream_start stream_id={stream_id} worker={worker_id} "
            f"text_len={len(text_to_speak)} preview={preview!r} text={text_to_speak!r} snap={snap}"
        )

    _ensure_model_warm(tts_model, model_idx=worker_id)
    max_new_tokens = _estimate_max_new_tokens(text_to_speak)
    if QWEN3_DIAG:
        print(
            f"[TTS_DIAG] stream_decode worker={worker_id} text_len={len(text_to_speak)} "
            f"max_new_tokens={max_new_tokens}"
        )

    chunk_size = max(1, int(DEFAULT_STREAM_PARAMS.get("emit_every_frames", 8)))

    chunk_iter = tts_model.generate_voice_clone_streaming(
        text=text_to_speak,
        language=QWEN3_LANGUAGE,
        ref_audio=QWEN3_REF_AUDIO,
        ref_text=QWEN3_REF_TEXT,
        do_sample=DEFAULT_PARAMS["do_sample"],
        temperature=DEFAULT_PARAMS["temperature"],
        top_p=DEFAULT_PARAMS["top_p"],
        top_k=DEFAULT_PARAMS["top_k"],
        repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        xvec_only=QWEN3_XVECTOR_ONLY,
    )
    chunk_count = 0
    total_pcm_bytes = 0
    last_stop_reason = None
    try:
        for wav_chunk, sr, timing in chunk_iter:
            chunk_count += 1
            if isinstance(timing, dict):
                last_stop_reason = timing.get("stop_reason", last_stop_reason)
            wav_chunk = np.asarray(wav_chunk, dtype=np.float32)
            if wav_chunk.size == 0:
                if QWEN3_DIAG:
                    print(
                        f"[TTS_DIAG] stream_chunk stream_id={stream_id} worker={worker_id} "
                        f"chunk={chunk_count} empty=1 timing={timing}"
                    )
                continue
            wav_chunk = _resample_if_needed(wav_chunk, int(sr), DEFAULT_PARAMS["target_sr"])
            pcm = _to_pcm16_bytes(wav_chunk)
            pcm_len = len(pcm) if pcm else 0
            total_pcm_bytes += pcm_len
            if QWEN3_STAGE_TIMING or QWEN3_DIAG:
                print(
                    f"[TTS_DIAG] stream_chunk stream_id={stream_id} worker={worker_id} "
                    f"chunk={chunk_count} wav_samples={int(wav_chunk.size)} pcm_bytes={pcm_len} timing={timing}"
                )
            if pcm:
                yield pcm
        if QWEN3_DIAG:
            print(
                f"[TTS_DIAG] stream_end stream_id={stream_id} worker={worker_id} "
                f"text_len={len(text_to_speak)} chunks={chunk_count} total_pcm_bytes={total_pcm_bytes} "
                f"last_stop_reason={last_stop_reason or 'unknown'}"
            )
    except Exception as e:
        print(
            f"[TTS_DIAG] stream_exception stream_id={stream_id} worker={worker_id} "
            f"text_len={len(text_to_speak)} chunks={chunk_count} total_pcm_bytes={total_pcm_bytes} err={e}"
        )
        raise


def synthesize_speech_to_memory_with_timing(text_to_speak: str):
    try:
        wav, sr, timings = _generate_wav_with_timing(text_to_speak, prompt_text=None)
        wav = _resample_if_needed(wav, sr, DEFAULT_PARAMS["target_sr"])
        pcm = _to_pcm16_bytes(wav)
        return pcm, timings
    except Exception as e:
        print(f"[ERROR] Memory synthesis with timing failed: {e}")
        traceback.print_exc()
        return None, None


try:
    print(f"[INFO] Faster Qwen3-TTS: loading {PARALLEL_TTS_MODEL_COUNT} model instance(s)...")
    for i in range(max(1, PARALLEL_TTS_MODEL_COUNT)):
        tts_model = FasterQwen3TTS.from_pretrained(
            model_name=QWEN3_MODEL_PATH,
            device=GLOBAL_DEVICE,
            dtype=_resolve_dtype(QWEN3_DTYPE),
            attn_implementation="eager",  # faster-qwen3-tts is designed without flash-attn dependency
        )

        # CUDA graph capture cannot include multinomial sampling.
        # Force deterministic predictor path even if upstream defaults to do_sample=True.
        if hasattr(tts_model, "predictor_graph") and tts_model.predictor_graph is not None:
            tts_model.predictor_graph.do_sample = False
        GLOBAL_TTS_MODELS.append(tts_model)
        print(f"[INFO] Faster Qwen3-TTS: loaded instance {i+1}/{PARALLEL_TTS_MODEL_COUNT}")

    GLOBAL_TTS_MODEL = GLOBAL_TTS_MODELS[0] if GLOBAL_TTS_MODELS else None

    if not QWEN3_REF_AUDIO:
        raise ValueError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if (not QWEN3_XVECTOR_ONLY) and (not QWEN3_REF_TEXT):
        raise ValueError("QWEN3_REF_TEXT is empty. Set transcript or enable QWEN3_XVECTOR_ONLY=1.")

    print("[INFO] Faster Qwen3-TTS: sequential warm-up for all model instances...")
    for i, tts_model in enumerate(GLOBAL_TTS_MODELS, start=1):
        _ensure_model_warm(tts_model, model_idx=i)
    print("[INFO] Faster Qwen3-TTS initialization complete.")
except Exception as e:
    print(f"[ERROR] Faster Qwen3-TTS init failed: {e}")
    traceback.print_exc()
    GLOBAL_TTS_MODEL = None
    GLOBAL_TTS_MODELS = []


if __name__ == "__main__":
    print("\n--- Faster Qwen3-TTS test ---")
    ok = synthesize_speech(
        "こんにちは。これはFaster Qwen3-TTSのテストです。",
        "faster_qwen3_test_output.wav",
    )
    print(f"test_result={ok}")
