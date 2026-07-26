import logging

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from redis import from_url
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from services.groq_client import transcribe_audio


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    database_status = "ok"
    redis_status = "ok"

    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database health check failed")
        database_status = "error"

    try:
        redis_client = from_url(settings.REDIS_URL)
        redis_client.ping()
    except Exception:
        logger.exception("Redis health check failed")
        redis_status = "error"

    return {
        "status": "ok" if database_status == "ok" and redis_status == "ok" else "error",
        "database": database_status,
        "redis": redis_status,
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            transcription = transcribe_audio(audio_bytes, "chunk.webm")

            if transcription is not None:
                await websocket.send_json({"text": transcription})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
