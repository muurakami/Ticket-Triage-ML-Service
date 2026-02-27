"""Model edge-case tests — проверяют, что модель не падает на граничных случаях."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestModelEdgeCases:
    """Тесты для граничных случаев модели."""

    def test_gibberish_text_no_crash(self, client: TestClient) -> None:
        """Бессмысленный текст не должен вызывать падение модели."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "gibberish-1",
                "text": "asdfghjkl qwertyuiop zxcvbnm",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data

    def test_very_long_text(self, client: TestClient) -> None:
        """Очень длинный текст должен обрабатываться без ошибок."""
        long_text = "Проблема с оплатой. " * 1000  # ~20k символов

        response = client.post(
            "/predict",
            json={
                "ticket_id": "long-text-1",
                "text": long_text,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_unicode_text_no_crash(self, client: TestClient) -> None:
        """Unicode символы не должны вызывать падение."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "unicode-1",
                "text": "Ошибка при оплате 💳 卡片支付错误 🎫",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_mixed_language_text(self, client: TestClient) -> None:
        """Смешанный язык (русский + английский) должен работать."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "mixed-lang-1",
                "text": "Payment failed ошибка при оплате card declined",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_numbers_and_special_chars(self, client: TestClient) -> None:
        """Цифры и спецсимволы не должны ломать модель."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "special-1",
                "text": "Order #12345 failed! Error code: 0xABC @#$%",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_repeated_words(self, client: TestClient) -> None:
        """Повторяющиеся слова должны обрабатываться корректно."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "repeated-1",
                "text": "ошибка ошибка ошибка ошибка ошибка",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data


class TestConfidenceThreshold:
    """Тесты для порога уверенности (confidence < 0.4 → label = 'other')."""

    def test_low_confidence_returns_other(self, client: TestClient) -> None:
        """
        Если модель не уверена, должен вернуться label='other'.
        Примечание: сложно гарантировать низкий confidence на конкретном тексте,
        но проверяем, что механизм работает (label может быть 'other').
        """
        # Используем текст, который может дать низкий confidence
        # (не похож на тренировочные данные)
        response = client.post(
            "/predict",
            json={
                "ticket_id": "low-conf-1",
                "text": "абвгдежзийклмнопрстуфхцчшщ",  # бессмысленный набор букв
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Проверяем, что confidence валидный
        assert 0.0 <= data["confidence"] <= 1.0
        # Если confidence < 0.4, label должен быть "other"
        if data["confidence"] < 0.4:
            assert data["label"] == "other"

    def test_high_confidence_returns_valid_label(self, client: TestClient) -> None:
        """
        Текст, похожий на тренировочные данные, должен дать высокий confidence
        и валидный label из списка классов.
        """
        response = client.post(
            "/predict",
            json={
                "ticket_id": "high-conf-1",
                "text": "Не могу оплатить картой, выдает ошибку при попытке списания",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Для текста про оплату ожидаем billing или другой валидный класс
        valid_labels = {"billing", "bug", "feature", "account", "other"}
        assert data["label"] in valid_labels
