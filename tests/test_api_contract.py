"""API contract tests — проверяют соответствие API контракту."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthzEndpoint:
    """Тесты для GET /healthz."""

    def test_healthz_returns_ok(self, client: TestClient) -> None:
        """Health check должен возвращать status=ok и model_loaded=true."""
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Тесты для POST /predict."""

    def test_predict_valid_request(self, client: TestClient) -> None:
        """Валидный запрос должен вернуть 200 и корректную структуру ответа."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "test-123",
                "text": "Не могу оплатить картой, выдает ошибку",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"] == "test-123"
        assert "label" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text_returns_422(self, client: TestClient) -> None:
        """Пустой текст должен вернуть 422 Unprocessable Entity."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "test-456",
                "text": "",
            },
        )

        assert response.status_code == 422

    def test_predict_short_text_returns_422(self, client: TestClient) -> None:
        """Текст короче 5 символов должен вернуть 422."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "test-789",
                "text": "ау",  # 2 символа
            },
        )

        assert response.status_code == 422

    def test_predict_whitespace_only_text_returns_422(self, client: TestClient) -> None:
        """Текст из одних пробелов должен вернуть 422."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "test-000",
                "text": "     ",  # только пробелы
            },
        )

        assert response.status_code == 422

    def test_predict_strips_whitespace(self, client: TestClient) -> None:
        """Пробелы по краям должны удаляться, но запрос проходить."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "test-strip",
                "text": "   Кнопка сохранить не работает в профиле   ",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"] == "test-strip"

    def test_predict_missing_ticket_id_returns_422(self, client: TestClient) -> None:
        """Отсутствие ticket_id должно вернуть 422."""
        response = client.post(
            "/predict",
            json={
                "text": "Нормальный текст тикета",
            },
        )

        assert response.status_code == 422

    def test_predict_missing_text_returns_422(self, client: TestClient) -> None:
        """Отсутствие text должно вернуть 422."""
        response = client.post(
            "/predict",
            json={
                "ticket_id": "no-text",
            },
        )

        assert response.status_code == 422
