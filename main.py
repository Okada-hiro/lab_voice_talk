from text_to_speech_with_openAI import stream_tts_openai


if __name__ == "__main__":
    stream_tts_openai(
        text="お電話ありがとうございます。事故のご連絡ですね。私が対応いたします。",
        model="gpt-4o-mini-tts",
        voice="marin",
        sample_rate=24000,
        speed=0.95,
        pitch=0.3,
        style="calm",
        output_path="openai_tts_out.wav",
    )
