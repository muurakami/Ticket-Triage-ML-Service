from __future__ import annotations

from ticket_triage.core.schemas import Label


def get_dummy_data() -> tuple[list[str], list[Label]]:
    """
    Генерирует фейковые данные для обучения baseline-модели.

    В реальности тут будет чтение из БД или CSV с помощью Pandas/Polars.
    Намеренно добавлены "грязные" записи для демонстрации валидации через TicketIn.
    """
    clean_data: list[tuple[str, Label]] = [
        ("Не могу оплатить картой, выдает ошибку", "billing"),
        ("Где скачать чек за прошлый месяц?", "billing"),
        ("Кнопка 'Сохранить' не нажимается в профиле", "bug"),
        ("Приложение вылетает при запуске", "bug"),
        ("Сделайте темную тему, глаза болят", "feature"),
        ("Хочу интеграцию с Telegram", "feature"),
        ("Как поменять пароль?", "account"),
        ("Заблокировали аккаунт, помогите", "account"),
    ] * 100

    # TicketIn должен их отловить и отклонить
    dirty_data: list[tuple[str, Label]] = [
        ("", "billing"),  # Пустая строка
        ("ау", "bug"),  # Слишком короткий текст (2 символа)
        ("   ", "feature"),  # Только пробелы — после strip() станет ""
        ("ok", "account"),  # 2 символа — меньше минимума
        ("баг", "bug"),  # 3 символа — меньше минимума
    ]

    data = clean_data + dirty_data  # type: ignore[operator]
    texts = [item[0] for item in data]
    labels: list[Label] = [item[1] for item in data]
    return texts, labels
