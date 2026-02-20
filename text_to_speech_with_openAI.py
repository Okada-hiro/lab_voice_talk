import argparse
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv


DEFAULT_TEXT = "お電話ありがとうございます。事故のご連絡ですね。私が対応いたします。"


def stream_tts_openai(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "marin",
    sample_rate: int = 24000,
    speed: float = 0.95,
    pitch: float = 0.3,
    style: str = "calm",
    output_path: str | None = "openai_tts_out.wav",
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it with: export OPENAI_API_KEY='your_api_key'"
        )

    client = OpenAI(api_key=api_key)

    stream = None
    can_play = True
    try:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        stream.start()
    except Exception as e:
        can_play = False
        print(f"[WARN] Audio device unavailable. Fallback to file output only: {e}")

    collected = []
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
                collected.append(audio.copy())
                if can_play and stream is not None:
                    stream.write(audio)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    if output_path and collected:
        pcm = np.concatenate(collected)
        sf.write(output_path, pcm, sample_rate, subtype="PCM_16")
        print(f"[INFO] Saved audio: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI TTS streaming playback")
    parser.add_argument("--text", default=os.getenv("OPENAI_TTS_TEXT", DEFAULT_TEXT))
    parser.add_argument("--model", default=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    parser.add_argument("--voice", default=os.getenv("OPENAI_TTS_VOICE", "marin"))
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("OPENAI_TTS_SAMPLE_RATE", "24000")))
    parser.add_argument("--speed", type=float, default=float(os.getenv("OPENAI_TTS_SPEED", "0.95")))
    parser.add_argument("--pitch", type=float, default=float(os.getenv("OPENAI_TTS_PITCH", "0.3")))
    parser.add_argument("--style", default=os.getenv("OPENAI_TTS_STYLE", "calm"))
    parser.add_argument("--output", default=os.getenv("OPENAI_TTS_OUTPUT", "openai_tts_out.wav"))
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
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
