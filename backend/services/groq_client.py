import logging

from groq import Groq

from config import settings


logger = logging.getLogger(__name__)
client = Groq(api_key=settings.GROQ_API_KEY)


def transcribe_audio(audio_bytes: bytes, filename: str) -> str | None:
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(filename, audio_bytes),
        )
        return transcription.text
    except Exception:
        logger.exception("Failed to transcribe audio with Groq")
        return None
