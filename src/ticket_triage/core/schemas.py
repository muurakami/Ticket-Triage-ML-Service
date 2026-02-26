from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

# Допустимые метки классификации тикетов
Label = Literal["billing", "bug", "feature", "account"]


class TicketIn(BaseModel):
    """Входящий тикет. Pydantic валидирует и очищает данные перед обучением."""

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
