import io
import os
import sys
import time
import traceback

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
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QWEN3_MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
QWEN3_REF_AUDIO = os.getenv("QWEN3_REF_AUDIO", "")
QWEN3_REF_TEXT = os.getenv("QWEN3_REF_TEXT", "")
QWEN3_LANGUAGE = os.getenv("QWEN3_LANGUAGE", "Japanese")
QWEN3_DTYPE = os.getenv("QWEN3_DTYPE", "bfloat16")
QWEN3_XVECTOR_ONLY = os.getenv("QWEN3_XVECTOR_ONLY", "1") == "1"
QWEN3_STAGE_TIMING = os.getenv("QWEN3_STAGE_TIMING", "0") == "1"

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


def _generate_wav(text_to_speak: str, prompt_text: str = None):
    if GLOBAL_TTS_MODEL is None:
        raise RuntimeError("Faster Qwen3-TTS model is not initialized.")
    if not QWEN3_REF_AUDIO:
        raise RuntimeError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if prompt_text:
        print("[WARN] prompt_text/instruct is ignored in voice-clone mode.")

    wavs, sr = GLOBAL_TTS_MODEL.generate_voice_clone(
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
        xvec_only=QWEN3_XVECTOR_ONLY,
    )
    if len(wavs) == 0:
        raise RuntimeError("No waveform was generated.")
    return np.asarray(wavs[0], dtype=np.float32), int(sr)


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
    if GLOBAL_TTS_MODEL is None:
        raise RuntimeError("Faster Qwen3-TTS model is not initialized.")
    if not QWEN3_REF_AUDIO:
        raise RuntimeError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if prompt_text:
        print("[WARN] prompt_text/instruct is ignored in streaming mode.")

    chunk_size = max(1, int(DEFAULT_STREAM_PARAMS.get("emit_every_frames", 8)))

    chunk_iter = GLOBAL_TTS_MODEL.generate_voice_clone_streaming(
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
    print("[INFO] Faster Qwen3-TTS: loading model...")
    GLOBAL_TTS_MODEL = FasterQwen3TTS.from_pretrained(
        model_name=QWEN3_MODEL_PATH,
        device=GLOBAL_DEVICE,
        dtype=_resolve_dtype(QWEN3_DTYPE),
        attn_implementation="eager",  # faster-qwen3-tts is designed without flash-attn dependency
    )

    if not QWEN3_REF_AUDIO:
        raise ValueError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")

    if (not QWEN3_XVECTOR_ONLY) and (not QWEN3_REF_TEXT):
        raise ValueError("QWEN3_REF_TEXT is empty. Set transcript or enable QWEN3_XVECTOR_ONLY=1.")

    print("[INFO] Faster Qwen3-TTS: warm-up inference...")
    _ = GLOBAL_TTS_MODEL.generate_voice_clone(
        text="あ",
        language=QWEN3_LANGUAGE,
        ref_audio=QWEN3_REF_AUDIO,
        ref_text=QWEN3_REF_TEXT,
        do_sample=False,
        max_new_tokens=96,
        xvec_only=QWEN3_XVECTOR_ONLY,
    )
    print("[INFO] Faster Qwen3-TTS initialization complete.")
except Exception as e:
    print(f"[ERROR] Faster Qwen3-TTS init failed: {e}")
    traceback.print_exc()
    GLOBAL_TTS_MODEL = None


if __name__ == "__main__":
    print("\n--- Faster Qwen3-TTS test ---")
    ok = synthesize_speech(
        "こんにちは。これはFaster Qwen3-TTSのテストです。",
        "faster_qwen3_test_output.wav",
    )
    print(f"test_result={ok}")
