from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ticket_triage.ml.train import train_baseline


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Обучает модель во временный файл один раз за сессию тестов.
    Использует tmp_path_factory для автоматической очистки.
    """
    output_path = tmp_path_factory.mktemp("models") / "model.joblib"
    train_baseline(output_path=output_path)
    return output_path


@pytest.fixture
def client(trained_model_path: Path) -> TestClient:
    """
    Создаёт TestClient с предзагруженной моделью.
    Обходит lifespan, чтобы избежать загрузки модели из DEFAULT_MODEL_PATH.
    """
    from ticket_triage.api.app import app
    from ticket_triage.ml.artifacts import load_artifacts

    # Загружаем артефакты напрямую в app.state
    app.state.artifacts = load_artifacts(trained_model_path)

    # Create TestClient without running lifespan startup
    # (raise_server_exceptions=True by default)
    # lifespan is bypassed since app.state.artifacts is already set
    with TestClient(app=app, raise_server_exceptions=True) as c:
        yield c
