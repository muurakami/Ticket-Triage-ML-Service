from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class SklearnPredictor(Protocol):
    """Минимальный интерфейс sklearn-совместимой модели классификации."""

    def predict(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...


@dataclass
class Artifacts:
    """Контейнер для всех артефактов обученной модели."""

    vectorizer: TfidfVectorizer
    model: SklearnPredictor
    labels: list[str]


def save_artifacts(path: Path, artifacts: Artifacts) -> None:
    """Сохраняет артефакты модели на диск через joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)


def load_artifacts(path: Path) -> Artifacts:
    """Загружает артефакты модели с диска."""
    if not path.exists():
        raise FileNotFoundError(
            f"Артефакты модели не найдены: {path}. "
            "Запустите `python -m ticket_triage.ml.train` для обучения."
        )
    result: Artifacts = joblib.load(path)
    return result
