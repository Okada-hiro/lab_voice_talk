import os
import io
import traceback

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


GLOBAL_TTS_MODEL = None
GLOBAL_VOICE_CLONE_PROMPT = None
GLOBAL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Environment-based configuration for easy deployment tuning.
QWEN3_MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
QWEN3_REF_AUDIO = os.getenv("QWEN3_REF_AUDIO", "")
QWEN3_REF_TEXT = os.getenv("QWEN3_REF_TEXT", "")
QWEN3_LANGUAGE = os.getenv("QWEN3_LANGUAGE", "Japanese")
QWEN3_DTYPE = os.getenv("QWEN3_DTYPE", "bfloat16")
QWEN3_XVECTOR_ONLY = os.getenv("QWEN3_XVECTOR_ONLY", "0") == "1"

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


try:
    print("[INFO] Qwen3-TTS: loading model...")
    GLOBAL_TTS_MODEL = Qwen3TTSModel.from_pretrained(
        QWEN3_MODEL_PATH,
        device_map=GLOBAL_DEVICE,
        dtype=_resolve_dtype(QWEN3_DTYPE),
        attn_implementation="flash_attention_2" if "cuda" in GLOBAL_DEVICE else None,
    )

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
