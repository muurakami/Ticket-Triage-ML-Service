from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Допустимые метки классификации тикетов
Label = Literal["billing", "bug", "feature", "account"]
# Метки для предсказания включают "other" для случаев низкой уверенности
PredictLabel = Label | Literal["other"]


class TicketIn(BaseModel):
    """Входящий тикет: валидация и очистка данных перед ML."""

    ticket_id: str
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_meaningful(cls, v: str) -> str:
        """Убираем лишние пробелы и проверяем минимальную длину."""
        cleaned = v.strip()
        if len(cleaned) < 5:
            raise ValueError(
                f"Текст тикета слишком короткий ({len(cleaned)} символов). "
                "Минимум 5 символов."
            )
        return cleaned


class PredictResponse(BaseModel):
    """Ответ API на запрос классификации тикета."""

    ticket_id: str
    label: PredictLabel  # "billing" | "bug" | "feature" | "account" | "other"
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Max probability from predict_proba"
    )
