from __future__ import annotations

import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ticket_triage.core.schemas import TicketIn
from ticket_triage.ml.artifacts import Artifacts, save_artifacts
from ticket_triage.ml.data import get_dummy_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Путь к артефактам относительно корня проекта, независимо от CWD при запуске
_PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "artifacts" / "model.joblib"


def train_baseline(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    logger.info("1. Загрузка и валидация данных...")
    raw_texts, raw_labels = get_dummy_data()

    # Валидируем данные через Pydantic перед обучением
    clean_texts: list[str] = []
    clean_labels: list[str] = []

    for i, (text, label) in enumerate(zip(raw_texts, raw_labels, strict=True)):
        try:
            # TicketIn сам почистит лишние пробелы через @field_validator
            ticket = TicketIn(ticket_id=str(i), text=text)
            clean_texts.append(ticket.text)
            clean_labels.append(label)
        except Exception as e:
            logger.warning("Пропущен битый тикет %d: текст=%r | ошибка: %s", i, text, e)

    logger.info("2. Разбиение на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        clean_texts, clean_labels, test_size=0.2, random_state=42
    )

    logger.info("3. Обучение векторизатора и модели...")
    # TF-IDF превращает текст в матрицу чисел; берём слова и биграммы
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    # Линейная модель, которая учится предсказывать класс по этой матрице
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    logger.info("4. Оценка качества (Evaluation)...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # classification_report покажет Precision, Recall и F1-score
    report = classification_report(y_test, y_pred)
    logger.info("Classification report:\n%s", report)

    logger.info("5. Упаковка и сохранение артефактов...")
    labels = list(model.classes_)

    artifacts = Artifacts(
        vectorizer=vectorizer,
        model=model,  # type: ignore[arg-type]  # SklearnPredictor protocol match
        labels=labels,
    )

    save_artifacts(output_path, artifacts)
    logger.info("Готово! Артефакты сохранены в %s", output_path.absolute())


if __name__ == "__main__":
    train_baseline()
