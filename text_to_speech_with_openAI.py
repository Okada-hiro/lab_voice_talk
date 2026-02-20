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
OPENAI_TTS_SAMPLE_RATE = int(os.getenv("OPENAI_TTS_SAMPLE_RATE", "24000"))
OPENAI_TTS_STYLE = os.getenv("OPENAI_TTS_STYLE", "calm")
OPENAI_TTS_BYTE_CHUNK = int(os.getenv("OPENAI_TTS_BYTE_CHUNK", "8192"))
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    (
        "polite Japanese customer support agent, "
        "calm, clear pronunciation, consistent speaking rate, "
        "slightly higher pitch than normal,"
        "slightly slow, neutral emotion"
    ),
)

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
    _ = OPENAI_TTS_STYLE
    instructions = prompt_text if prompt_text else OPENAI_TTS_INSTRUCTIONS

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text_to_speak,
        instructions=instructions,
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
            # PCM16は2バイト境界が必須。奇数バイトは次チャンクへ持ち越す。
            while len(pending) >= OPENAI_TTS_BYTE_CHUNK:
                candidate = OPENAI_TTS_BYTE_CHUNK
                if candidate % 2 == 1:
                    candidate -= 1
                out = bytes(pending[:candidate])
                del pending[:candidate]
                yield out

        # 末尾も2バイト境界までのみ返す（端数1バイトは破棄）
        tail_usable = len(pending) - (len(pending) % 2)
        if tail_usable > 0:
            yield bytes(pending[:tail_usable])

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
        joined = b"".join(parts)
        usable = len(joined) - (len(joined) % 2)
        return joined[:usable]
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
        pcm = pcm[: len(pcm) - (len(pcm) % 2)]
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
