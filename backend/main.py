import json
import logging

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from redis import from_url
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from agents.quality import quality_agent
from services.groq_client import transcribe_audio
from workflows.call_flow import call_graph


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _default_call_state(call_id: str) -> dict[str, object]:
    return {
        "call_id": call_id,
        "transcript_segment": "",
        "transcript": "",
        "intent": None,
        "intent_confidence": None,
        "low_confidence": None,
        "kb_suggestions": [],
        "suggested_response": None,
        "escalation_risk": None,
        "escalation_alert": None,
        "sentiment_history": [],
        "turn_count": 0,
        "last_segment_timestamp": None,
        "hold_time_proxy": 0.0,
    }


async def _load_call_state(call_id: str) -> dict[str, object]:
    key = f"call_session:{call_id}"
    try:
        raw_state = await redis_client.get(key)
    except Exception:
        logger.exception("Failed to read call session state for %s", call_id)
        return _default_call_state(call_id)

    if not raw_state:
        return _default_call_state(call_id)

    try:
        state = json.loads(raw_state)
        if not isinstance(state, dict):
            raise ValueError("call session state must be a JSON object")
        return state
    except Exception:
        logger.exception("Failed to decode call session state for %s", call_id)
        return _default_call_state(call_id)


async def _save_call_state(call_id: str, state: dict[str, object]) -> None:
    key = f"call_session:{call_id}"
    try:
        await redis_client.set(key, json.dumps(state), ex=3600)
    except Exception:
        logger.exception("Failed to persist call session state for %s", call_id)


def _call_update_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "transcript": state.get("transcript"),
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "kb_suggestions": state.get("kb_suggestions"),
        "suggested_response": state.get("suggested_response"),
        "escalation_risk": state.get("escalation_risk"),
        "escalation_alert": state.get("escalation_alert"),
    }


@app.get("/calls/{call_id}/qa_score")
async def get_qa_score(call_id: str):
    state = await _load_call_state(call_id)
    transcript = str(state.get("transcript") or "").strip()

    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="No call found for this call_id, or the call has no transcript yet",
        )

    try:
        result = await quality_agent.score_call(
            transcript=transcript,
            agent_id=settings.DEFAULT_AGENT_ID,
        )
        return result
    except Exception:
        logger.exception("QA scoring failed for call_id=%s", call_id)
        raise HTTPException(status_code=500, detail="QA scoring failed")


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


@app.websocket("/ws/calls/{call_id}")
async def websocket_call_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()

    state = await _load_call_state(call_id)
    await _save_call_state(call_id, state)

    while True:
        try:
            audio_bytes = await websocket.receive_bytes()
            transcription = transcribe_audio(audio_bytes, "chunk.webm")

            if not transcription:
                await websocket.send_json({"type": "error", "message": "transcription failed"})
                continue

            current_state = await _load_call_state(call_id)
            current_state["transcript_segment"] = transcription

            updated_state = await call_graph.ainvoke(current_state)
            await _save_call_state(call_id, updated_state)

            await websocket.send_json(
                {
                    "type": "update",
                    "data": _call_update_payload(updated_state),
                }
            )
        except WebSocketDisconnect:
            logger.info("Call WebSocket client disconnected for call_id=%s", call_id)
            break
        except Exception:
            logger.exception("Failed to process call websocket turn for call_id=%s", call_id)
