#今はこれ！ 2月20日 一旦安定する
import os
import io
import traceback
import threading
import time

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


GLOBAL_TTS_MODEL = None
GLOBAL_VOICE_CLONE_PROMPT = None
GLOBAL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GLOBAL_TTS_TIMING_LOCK = threading.Lock()

# Environment-based configuration for easy deployment tuning.
QWEN3_MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
QWEN3_REF_AUDIO = os.getenv("QWEN3_REF_AUDIO", "")
QWEN3_REF_TEXT = os.getenv("QWEN3_REF_TEXT", "")
QWEN3_LANGUAGE = os.getenv("QWEN3_LANGUAGE", "Japanese")
QWEN3_DTYPE = os.getenv("QWEN3_DTYPE", "bfloat16")
QWEN3_XVECTOR_ONLY = os.getenv("QWEN3_XVECTOR_ONLY", "0") == "1"
QWEN3_STAGE_TIMING = os.getenv("QWEN3_STAGE_TIMING", "0") == "1"
QWEN3_USE_TORCH_COMPILE = os.getenv("QWEN3_USE_TORCH_COMPILE", "0") == "1"
QWEN3_TORCH_COMPILE_MODE = os.getenv("QWEN3_TORCH_COMPILE_MODE", "reduce-overhead")
QWEN3_TORCH_COMPILE_DYNAMIC = os.getenv("QWEN3_TORCH_COMPILE_DYNAMIC", "0") == "1"
QWEN3_QUANT_MODE = os.getenv("QWEN3_QUANT_MODE", "none").strip().lower()
QWEN3_QUANT_TARGETS = os.getenv("QWEN3_QUANT_TARGETS", "talker,code_predictor")

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

DEFAULT_STREAM_PARAMS = {
    "emit_every_frames": 8,
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


def _resolve_quant_targets():
    targets = []
    for raw in QWEN3_QUANT_TARGETS.split(","):
        key = raw.strip().lower()
        if key:
            targets.append(key)
    return targets


def _find_quant_modules(tts_model):
    modules = []
    if not hasattr(tts_model, "model"):
        return modules

    base = tts_model.model
    for target in _resolve_quant_targets():
        if target == "talker" and hasattr(base, "talker"):
            modules.append(("talker", base.talker))
        elif target == "code_predictor" and hasattr(base, "code_predictor"):
            modules.append(("code_predictor", base.code_predictor))
        elif target == "model":
            modules.append(("model", base))
    return modules


def _apply_quantization_if_enabled(tts_model):
    if QWEN3_QUANT_MODE in ("", "none", "off", "false", "0"):
        print("[INFO] Qwen3-TTS quantization: disabled (QWEN3_QUANT_MODE=none).")
        return

    if QWEN3_QUANT_MODE != "int8":
        print(f"[WARN] Qwen3-TTS quantization: unknown mode '{QWEN3_QUANT_MODE}', skip.")
        return

    quant_modules = _find_quant_modules(tts_model)
    if not quant_modules:
        print("[WARN] Qwen3-TTS quantization: no target modules found, skip.")
        return

    try:
        from torchao.quantization import int8_weight_only, quantize_
    except Exception as e:
        print(f"[WARN] Qwen3-TTS quantization: torchao unavailable, skip int8 ({e}).")
        return

    print(f"[INFO] Qwen3-TTS quantization: mode=int8 targets={','.join(name for name, _ in quant_modules)}")
    ok_count = 0
    for name, module in quant_modules:
        try:
            quantize_(module, int8_weight_only())
            ok_count += 1
            print(f"[INFO] Qwen3-TTS quantization: applied int8 to '{name}'.")
        except Exception as e:
            print(f"[WARN] Qwen3-TTS quantization: failed on '{name}', keep fp path ({e}).")
    print(f"[INFO] Qwen3-TTS quantization: completed {ok_count}/{len(quant_modules)} targets.")


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
        raise RuntimeError("Qwen3-TTS model is not initialized.")
    if GLOBAL_VOICE_CLONE_PROMPT is None:
        raise RuntimeError("Voice clone prompt is not initialized. Check QWEN3_REF_AUDIO/QWEN3_REF_TEXT.")

    instruct = prompt_text if prompt_text else DEFAULT_PARAMS["instruct"]
    wavs, sr = GLOBAL_TTS_MODEL.generate_voice_clone(
        text=text_to_speak,
        language=QWEN3_LANGUAGE,
        instruct=instruct,
        voice_clone_prompt=GLOBAL_VOICE_CLONE_PROMPT,
        do_sample=DEFAULT_PARAMS["do_sample"],
        temperature=DEFAULT_PARAMS["temperature"],
        top_p=DEFAULT_PARAMS["top_p"],
        top_k=DEFAULT_PARAMS["top_k"],
        repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
        max_new_tokens=DEFAULT_PARAMS["max_new_tokens"],
    )
    if len(wavs) == 0:
        raise RuntimeError("No waveform was generated.")
    return np.asarray(wavs[0], dtype=np.float32), int(sr)


def _generate_wav_with_timing(text_to_speak: str, prompt_text: str = None):
    """
    Return (wav, sr, timings_ms) where timings_ms includes:
      - generate_ms: talker code generation
      - decode_ms: codec-to-waveform decode
      - total_ms: whole generate_voice_clone call
    """
    if GLOBAL_TTS_MODEL is None:
        raise RuntimeError("Qwen3-TTS model is not initialized.")
    if GLOBAL_VOICE_CLONE_PROMPT is None:
        raise RuntimeError("Voice clone prompt is not initialized. Check QWEN3_REF_AUDIO/QWEN3_REF_TEXT.")

    timings = {"generate_ms": None, "decode_ms": None, "total_ms": None}
    instruct = prompt_text if prompt_text else DEFAULT_PARAMS["instruct"]

    model_generate = GLOBAL_TTS_MODEL.model.generate
    tokenizer_decode = GLOBAL_TTS_MODEL.model.speech_tokenizer.decode

    def timed_model_generate(*args, **kwargs):
        t0 = time.perf_counter()
        out = model_generate(*args, **kwargs)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def timed_tokenizer_decode(*args, **kwargs):
        t0 = time.perf_counter()
        out = tokenizer_decode(*args, **kwargs)
        timings["decode_ms"] = (time.perf_counter() - t0) * 1000
        return out

    with GLOBAL_TTS_TIMING_LOCK:
        GLOBAL_TTS_MODEL.model.generate = timed_model_generate
        GLOBAL_TTS_MODEL.model.speech_tokenizer.decode = timed_tokenizer_decode
        try:
            t0 = time.perf_counter()
            wavs, sr = GLOBAL_TTS_MODEL.generate_voice_clone(
                text=text_to_speak,
                language=QWEN3_LANGUAGE,
                instruct=instruct,
                voice_clone_prompt=GLOBAL_VOICE_CLONE_PROMPT,
                do_sample=DEFAULT_PARAMS["do_sample"],
                temperature=DEFAULT_PARAMS["temperature"],
                top_p=DEFAULT_PARAMS["top_p"],
                top_k=DEFAULT_PARAMS["top_k"],
                repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
                max_new_tokens=DEFAULT_PARAMS["max_new_tokens"],
            )
            timings["total_ms"] = (time.perf_counter() - t0) * 1000
        finally:
            GLOBAL_TTS_MODEL.model.generate = model_generate
            GLOBAL_TTS_MODEL.model.speech_tokenizer.decode = tokenizer_decode

    if len(wavs) == 0:
        raise RuntimeError("No waveform was generated.")
    return np.asarray(wavs[0], dtype=np.float32), int(sr), timings


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
    """
    Yield PCM16 bytes chunks in streaming mode.
    Requires qwen_tts implementation that supports stream_generate_voice_clone().
    """
    if GLOBAL_TTS_MODEL is None:
        raise RuntimeError("Qwen3-TTS model is not initialized.")
    if GLOBAL_VOICE_CLONE_PROMPT is None:
        raise RuntimeError("Voice clone prompt is not initialized. Check QWEN3_REF_AUDIO/QWEN3_REF_TEXT.")
    if not hasattr(GLOBAL_TTS_MODEL, "stream_generate_voice_clone"):
        raise RuntimeError("Installed qwen_tts does not support stream_generate_voice_clone().")

    if prompt_text:
        # Current streaming fork does not accept instruct in stream_generate_voice_clone.
        print("[WARN] prompt_text/instruct is ignored in streaming mode.")

    chunk_iter = GLOBAL_TTS_MODEL.stream_generate_voice_clone(
        text=text_to_speak,
        language=QWEN3_LANGUAGE,
        voice_clone_prompt=GLOBAL_VOICE_CLONE_PROMPT,
        emit_every_frames=DEFAULT_STREAM_PARAMS["emit_every_frames"],
        decode_window_frames=DEFAULT_STREAM_PARAMS["decode_window_frames"],
        overlap_samples=DEFAULT_STREAM_PARAMS["overlap_samples"],
        max_frames=DEFAULT_STREAM_PARAMS["max_frames"],
        first_chunk_emit_every=DEFAULT_STREAM_PARAMS["first_chunk_emit_every"],
        first_chunk_decode_window=DEFAULT_STREAM_PARAMS["first_chunk_decode_window"],
        first_chunk_frames=DEFAULT_STREAM_PARAMS["first_chunk_frames"],
        repetition_penalty=DEFAULT_STREAM_PARAMS["repetition_penalty"],
        repetition_penalty_window=DEFAULT_STREAM_PARAMS["repetition_penalty_window"],
        use_optimized_decode=DEFAULT_STREAM_PARAMS["use_optimized_decode"],
        do_sample=DEFAULT_PARAMS["do_sample"],
        temperature=DEFAULT_PARAMS["temperature"],
        top_p=DEFAULT_PARAMS["top_p"],
        top_k=DEFAULT_PARAMS["top_k"],
    )

    chunk_idx = 0
    while True:
        model_wait_start = time.perf_counter()
        try:
            wav_chunk, sr = next(chunk_iter)
        except StopIteration:
            break
        model_wait_ms = (time.perf_counter() - model_wait_start) * 1000.0

        chunk_idx += 1
        copy_start = time.perf_counter()
        wav_chunk = np.asarray(wav_chunk, dtype=np.float32)
        copy_ms = (time.perf_counter() - copy_start) * 1000.0
        if wav_chunk.size == 0:
            if QWEN3_STAGE_TIMING:
                print(
                    f"[TTS_STAGE] chunk={chunk_idx} model_wait_ms={model_wait_ms:.1f} "
                    f"copy_ms={copy_ms:.1f} empty=1"
                )
            continue

        resample_start = time.perf_counter()
        wav_chunk = _resample_if_needed(wav_chunk, int(sr), DEFAULT_PARAMS["target_sr"])
        resample_ms = (time.perf_counter() - resample_start) * 1000.0

        pcm_start = time.perf_counter()
        pcm = _to_pcm16_bytes(wav_chunk)
        pcm_ms = (time.perf_counter() - pcm_start) * 1000.0

        if QWEN3_STAGE_TIMING:
            print(
                f"[TTS_STAGE] chunk={chunk_idx} model_wait_ms={model_wait_ms:.1f} "
                f"copy_ms={copy_ms:.1f} resample_ms={resample_ms:.1f} "
                f"pcm_ms={pcm_ms:.1f} sr_in={int(sr)} sr_out={DEFAULT_PARAMS['target_sr']} "
                f"pcm_bytes={len(pcm) if pcm else 0}"
            )

        if pcm:
            yield pcm


def synthesize_speech_to_memory_with_timing(text_to_speak: str):
    """
    Return (pcm16_bytes, timings_ms_dict) or (None, None) on failure.
    """
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
    print("[INFO] Qwen3-TTS: loading model...")
    GLOBAL_TTS_MODEL = Qwen3TTSModel.from_pretrained(
        QWEN3_MODEL_PATH,
        device_map=GLOBAL_DEVICE,
        dtype=_resolve_dtype(QWEN3_DTYPE),
        attn_implementation="flash_attention_2" if "cuda" in GLOBAL_DEVICE else None,
    )

    _apply_quantization_if_enabled(GLOBAL_TTS_MODEL)

    if QWEN3_USE_TORCH_COMPILE:
        if not hasattr(torch, "compile"):
            print("[WARN] QWEN3_USE_TORCH_COMPILE=1 but torch.compile is unavailable on this PyTorch.")
        else:
            try:
                # Prioritize compiling talker module used in generation path.
                if hasattr(GLOBAL_TTS_MODEL, "model") and hasattr(GLOBAL_TTS_MODEL.model, "talker"):
                    print(
                        "[INFO] Qwen3-TTS: enabling torch.compile on talker "
                        f"(mode={QWEN3_TORCH_COMPILE_MODE}, dynamic={QWEN3_TORCH_COMPILE_DYNAMIC})..."
                    )
                    GLOBAL_TTS_MODEL.model.talker = torch.compile(
                        GLOBAL_TTS_MODEL.model.talker,
                        mode=QWEN3_TORCH_COMPILE_MODE,
                        dynamic=QWEN3_TORCH_COMPILE_DYNAMIC,
                    )
                    print("[INFO] Qwen3-TTS: torch.compile enabled on talker.")
                elif hasattr(GLOBAL_TTS_MODEL, "model"):
                    print(
                        "[INFO] Qwen3-TTS: enabling torch.compile on root model "
                        f"(mode={QWEN3_TORCH_COMPILE_MODE}, dynamic={QWEN3_TORCH_COMPILE_DYNAMIC})..."
                    )
                    GLOBAL_TTS_MODEL.model = torch.compile(
                        GLOBAL_TTS_MODEL.model,
                        mode=QWEN3_TORCH_COMPILE_MODE,
                        dynamic=QWEN3_TORCH_COMPILE_DYNAMIC,
                    )
                    print("[INFO] Qwen3-TTS: torch.compile enabled on root model.")
                else:
                    print("[WARN] Qwen3-TTS: compile target not found, skip torch.compile.")
            except Exception as compile_e:
                print(f"[WARN] Qwen3-TTS: torch.compile failed, continue without compile: {compile_e}")

    if not QWEN3_REF_AUDIO:
        raise ValueError("QWEN3_REF_AUDIO is empty. Set reference audio path for voice cloning.")
    if (not QWEN3_XVECTOR_ONLY) and (not QWEN3_REF_TEXT):
        raise ValueError("QWEN3_REF_TEXT is empty. Set transcript or enable QWEN3_XVECTOR_ONLY=1.")

    print("[INFO] Qwen3-TTS: creating reusable voice clone prompt...")
    GLOBAL_VOICE_CLONE_PROMPT = GLOBAL_TTS_MODEL.create_voice_clone_prompt(
        ref_audio=QWEN3_REF_AUDIO,
        ref_text=QWEN3_REF_TEXT if QWEN3_REF_TEXT else None,
        x_vector_only_mode=QWEN3_XVECTOR_ONLY,
    )

    print("[INFO] Qwen3-TTS: warm-up inference...")
    _ = GLOBAL_TTS_MODEL.generate_voice_clone(
        text="あ",
        language=QWEN3_LANGUAGE,
        instruct=DEFAULT_PARAMS["instruct"],
        voice_clone_prompt=GLOBAL_VOICE_CLONE_PROMPT,
        do_sample=False,
        max_new_tokens=96,
    )
    print("[INFO] Qwen3-TTS initialization complete.")
except Exception as e:
    print(f"[ERROR] Qwen3-TTS init failed: {e}")
    traceback.print_exc()
    GLOBAL_TTS_MODEL = None
    GLOBAL_VOICE_CLONE_PROMPT = None


if __name__ == "__main__":
    print("\n--- Qwen3-TTS new_text_to_speech test ---")
    ok = synthesize_speech(
        "こんにちは。これはQwen3-TTSのテストです。",
        "qwen3_test_output.wav",
    )
    print(f"test_result={ok}")
