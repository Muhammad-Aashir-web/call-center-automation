"""Scratch script: generate two short WAV audio clips using Windows' offline
TTS voice, for manually testing /ws/calls/{call_id}. Delete after use."""

import pyttsx3

SENTENCES = [
    ("_test_audio_1.wav", "My internet has been down for two days and nobody has helped me."),
    ("_test_audio_2.wav", "This is extremely frustrating and I am ready to cancel my service."),
]


def main() -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)

    for filename, text in SENTENCES:
        engine.save_to_file(text, filename)

    engine.runAndWait()
    print("Generated:", [f for f, _ in SENTENCES])


if __name__ == "__main__":
    main()