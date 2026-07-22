import logging

from fastapi import Depends, FastAPI
from redis import from_url
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from database import get_db


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
