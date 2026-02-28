from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from ticket_triage.core.schemas import PredictResponse, TicketIn
from ticket_triage.ml.artifacts import Artifacts, load_artifacts

# Configure structlog for JSON output (K8s/ELK compatible)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Путь к артефактам относительно корня проекта
_PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "model.joblib"

# Порог уверенности: если confidence < 0.4, возвращаем "other"
CONFIDENCE_THRESHOLD = 0.4

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
INFERENCE_COUNT = Counter(
    "inference_total",
    "Total inference requests",
    ["label"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Model inference duration in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель один раз при старте сервера (Singleton pattern)."""
    # Пропускаем загрузку, если артефакты уже установлены (например, в тестах)
    if hasattr(app.state, "artifacts"):
        logger.info("Artifacts already loaded, skipping lifespan load")
        yield
        return

    try:
        artifacts = load_artifacts(DEFAULT_MODEL_PATH)
        app.state.artifacts = artifacts
        logger.info("model_loaded", labels_count=len(artifacts.labels))
    except FileNotFoundError as e:
        logger.error("model_load_failed", error=str(e))
        app.state.artifacts = None
    yield


app = FastAPI(
    title="Ticket Triage ML Service",
    description="Классификация тикетов поддержки с помощью TF-IDF + LogisticRegression",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Middleware for Prometheus metrics."""
    start_time = time.perf_counter()

    response = await call_next(request)

    latency = time.perf_counter() - start_time
    method = request.method
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    return response


@app.get("/healthz")
async def healthz() -> dict[str, str | bool]:
    """Health check endpoint. Возвращает статус и признак загруженной модели."""
    model_loaded = hasattr(app.state, "artifacts") and app.state.artifacts is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


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
        "predict_request",
        ticket_id=request.ticket_id,
        text_length=len(request.text),
    )

    artifacts: Artifacts = app.state.artifacts

    # Guard: ensure model is loaded
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    # Track inference time
    inference_start = time.perf_counter()

    # Векторизация текста
    X = artifacts.vectorizer.transform([request.text])

    # Предсказание вероятностей
    probas = artifacts.model.predict_proba(X)[0]
    max_idx = probas.argmax()
    confidence = float(probas[max_idx])
    label = artifacts.labels[max_idx]

    inference_latency = time.perf_counter() - inference_start
    INFERENCE_LATENCY.observe(inference_latency)

    # Порог уверенности
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "low_confidence",
            ticket_id=request.ticket_id,
            confidence=confidence,
            forced_label="other",
        )
        label = "other"

    # Track inference count by label
    INFERENCE_COUNT.labels(label=label).inc()

    logger.info(
        "predict_result",
        ticket_id=request.ticket_id,
        label=label,
        confidence=confidence,
        inference_latency_seconds=inference_latency,
    )

    return PredictResponse(
        ticket_id=request.ticket_id,
        label=label,
        confidence=confidence,
    )
