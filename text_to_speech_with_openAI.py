import io
import os
import traceback
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
OPENAI_TTS_RESPONSE_FORMAT = os.getenv("OPENAI_TTS_RESPONSE_FORMAT", "pcm")
OPENAI_TTS_SAMPLE_RATE = int(os.getenv("OPENAI_TTS_SAMPLE_RATE", "16000"))
OPENAI_TTS_STYLE = os.getenv("OPENAI_TTS_STYLE", "calm")
OPENAI_TTS_BYTE_CHUNK = int(os.getenv("OPENAI_TTS_BYTE_CHUNK", "4096"))

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it with: export OPENAI_API_KEY='your_api_key'"
        )
    _client = OpenAI(api_key=api_key)
    return _client


def _iter_openai_pcm_chunks(text_to_speak: str, prompt_text: str = None) -> Iterator[bytes]:
    client = _get_client()
    _ = prompt_text if prompt_text else OPENAI_TTS_STYLE

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text_to_speak,
        response_format=OPENAI_TTS_RESPONSE_FORMAT,
    ) as response:
        for chunk in response.iter_bytes():
            if chunk:
                yield chunk


def synthesize_speech_to_memory_stream(text_to_speak: str, prompt_text: str = None):
    """
    new_text_to_speech.py と同じ役割:
    PCM16 bytes を逐次 yield する。
    """
    try:
        pending = bytearray()
        for chunk in _iter_openai_pcm_chunks(text_to_speak, prompt_text=prompt_text):
            pending.extend(chunk)
            while len(pending) >= OPENAI_TTS_BYTE_CHUNK:
                out = bytes(pending[:OPENAI_TTS_BYTE_CHUNK])
                del pending[:OPENAI_TTS_BYTE_CHUNK]
                yield out

        if pending:
            yield bytes(pending)

    except Exception as e:
        print(f"[ERROR] OpenAI stream synthesis failed: {e}")
        traceback.print_exc()
        return


def synthesize_speech_to_memory(text_to_speak: str) -> Optional[bytes]:
    """
    new_text_to_speech.py と同じ役割:
    一括PCM16 bytesを返す。
    """
    try:
        parts = []
        for chunk in _iter_openai_pcm_chunks(text_to_speak, prompt_text=None):
            parts.append(chunk)
        return b"".join(parts)
    except Exception as e:
        print(f"[ERROR] OpenAI memory synthesis failed: {e}")
        traceback.print_exc()
        return None


def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None) -> bool:
    """
    new_text_to_speech.py と同じ役割:
    音声ファイル保存。
    """
    try:
        pcm = b"".join(_iter_openai_pcm_chunks(text_to_speak, prompt_text=prompt_text))
        if not pcm:
            raise RuntimeError("No audio bytes were generated.")

        audio_i16 = np.frombuffer(pcm, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
        sf.write(output_wav_path, audio_f32, OPENAI_TTS_SAMPLE_RATE)
        print(f"[SUCCESS] Saved to {output_wav_path}")
        return True
    except Exception as e:
        print(f"[ERROR] OpenAI file synthesis failed: {e}")
        traceback.print_exc()
        return False
