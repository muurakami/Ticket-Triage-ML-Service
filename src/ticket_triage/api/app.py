from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ticket_triage.core.schemas import PredictResponse, TicketIn
from ticket_triage.ml.artifacts import Artifacts, load_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Путь к артефактам относительно корня проекта
_PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "model.joblib"

# Порог уверенности: если confidence < 0.4, возвращаем "other"
CONFIDENCE_THRESHOLD = 0.4


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель один раз при старте сервера (Singleton pattern)."""
    # Пропускаем загрузку, если артефакты уже установлены (например, в тестах)
    if hasattr(app.state, "artifacts"):
        logger.info("Artifacts already loaded, skipping lifespan load")
        yield
        return

    artifacts = load_artifacts(DEFAULT_MODEL_PATH)
    app.state.artifacts = artifacts
    logger.info("Model loaded: %d labels", len(artifacts.labels))
    yield


app = FastAPI(
    title="Ticket Triage ML Service",
    description="Классификация тикетов поддержки с помощью TF-IDF + LogisticRegression",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str | bool]:
    """Health check endpoint. Возвращает статус и признак загруженной модели."""
    return {"status": "ok", "model_loaded": hasattr(app.state, "artifacts")}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: TicketIn) -> PredictResponse:
    """
    Классифицирует тикет по тексту.

    - Валидирует вход через TicketIn (минимум 5 символов, strip whitespace)
    - Логирует запрос
    - Возвращает предсказанный label и confidence
    - Если confidence < 0.4, label = "other"
    """
    logger.info(
        "Predict request: ticket_id=%s, text_length=%d",
        request.ticket_id,
        len(request.text),
    )

    artifacts: Artifacts = app.state.artifacts

    # Guard: ensure model is loaded
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    # Векторизация текста
    X = artifacts.vectorizer.transform([request.text])

    # Предсказание вероятностей
    probas = artifacts.model.predict_proba(X)[0]
    max_idx = probas.argmax()
    confidence = float(probas[max_idx])
    label = artifacts.labels[max_idx]

    # Порог уверенности
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Low confidence %.3f for ticket_id=%s, forcing label='other'",
            confidence,
            request.ticket_id,
        )
        label = "other"

    logger.info(
        "Predict result: ticket_id=%s, label=%s, confidence=%.3f",
        request.ticket_id,
        label,
        confidence,
    )

    return PredictResponse(
        ticket_id=request.ticket_id,
        label=label,
        confidence=confidence,
    )
