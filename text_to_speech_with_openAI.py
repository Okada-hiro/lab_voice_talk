import argparse
import os

import numpy as np
import sounddevice as sd
from openai import OpenAI


DEFAULT_TEXT = "お電話ありがとうございます。事故のご連絡ですね。私が対応いたします。"


def stream_tts_openai(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "marin",
    sample_rate: int = 24000,
    speed: float = 0.95,
    pitch: float = 0.3,
    style: str = "calm",
) -> None:
    client = OpenAI()

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    stream.start()

    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            format="pcm16",
            sample_rate=sample_rate,
            speed=speed,
            pitch=pitch,
            style=style,
        ) as response:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                audio = np.frombuffer(chunk, dtype=np.int16)
                stream.write(audio)
    finally:
        stream.stop()
        stream.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI TTS streaming playback")
    parser.add_argument("--text", default=os.getenv("OPENAI_TTS_TEXT", DEFAULT_TEXT))
    parser.add_argument("--model", default=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    parser.add_argument("--voice", default=os.getenv("OPENAI_TTS_VOICE", "marin"))
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("OPENAI_TTS_SAMPLE_RATE", "24000")))
    parser.add_argument("--speed", type=float, default=float(os.getenv("OPENAI_TTS_SPEED", "0.95")))
    parser.add_argument("--pitch", type=float, default=float(os.getenv("OPENAI_TTS_PITCH", "0.3")))
    parser.add_argument("--style", default=os.getenv("OPENAI_TTS_STYLE", "calm"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stream_tts_openai(
        text=args.text,
        model=args.model,
        voice=args.voice,
        sample_rate=args.sample_rate,
        speed=args.speed,
        pitch=args.pitch,
        style=args.style,
    )


if __name__ == "__main__":
    main()
