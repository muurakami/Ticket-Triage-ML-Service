# Ticket Triage ML Service

ML-сервис для классификации тикетов поддержки с помощью TF-IDF + LogisticRegression.

## Быстрый старт

```bash
# Установка зависимостей
uv sync

# Обучение модели
python -m ticket_triage.ml.train

# Запуск сервера
uv run uvicorn ticket_triage.api.app:app --reload
```

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/predict` | POST | Классифицировать тикет |
| `/healthz` | GET | Health check |
| `/metrics` | GET | Prometheus метрики |

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "123", "text": "Не могу оплатить картой"}'
```

Ответ:
```json
{
  "ticket_id": "123",
  "label": "billing",
  "confidence": 0.92
}
```

### Confidence Threshold

Если `confidence < 0.4`, возвращается `label = "other"` как fallback.

## Архитектура

```
src/ticket_triage/
├── api/app.py          # FastAPI приложение
├── core/schemas.py    # Pydantic модели
└── ml/
    ├── artifacts.py   # Загрузка/сохранение модели
    ├── train.py       # Обучение
    └── data.py        # Данные для обучения
```

## Тесты

```bash
uv run pytest
```

- `tests/test_api_contract.py` — API контракт
- `tests/test_model_edge_cases.py` — Edge cases модели

## Технологии

- **FastAPI** — веб-фреймворк
- **scikit-learn** — TF-IDF + LogisticRegression
- **Pydantic** — валидация
- **structlog** — структурированное логирование (JSON для K8s/ELK)
- **Prometheus** — метрики

## Требования

- Python 3.13+
- uv (package manager)
