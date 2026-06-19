import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

_MOCK_PREDICTION = {
    "ticker": "AAPL",
    "prediction_date": "2024-12-20",
    "predicted_return": 0.012,
    "predicted_return_pct": "+1.200%",
    "predicted_direction": "UP",
    "confidence": 0.24,
    "current_price": 195.50,
}


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "available_tickers" in data
        assert "version" in data


class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_service_info(self):
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert "docs" in data


class TestPredictEndpoint:

    def test_invalid_ticker_rejected(self):
        response = client.post("/predict", json={"ticker": "INVALID123TOOLONG"})
        assert response.status_code == 422

    def test_empty_ticker_rejected(self):
        response = client.post("/predict", json={"ticker": ""})
        assert response.status_code == 422

    @patch("app.main.predict_for_ticker")
    def test_successful_prediction(self, mock_predict):
        mock_predict.return_value = _MOCK_PREDICTION
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["ticker"] == "AAPL"

    @patch("app.main.predict_for_ticker")
    def test_model_not_found_returns_404(self, mock_predict):
        mock_predict.side_effect = FileNotFoundError("Model not found")
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 404

    @patch("app.main.predict_for_ticker")
    def test_data_unavailable_returns_503(self, mock_predict):
        mock_predict.side_effect = ValueError("No data for ticker")
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 503


class TestBatchEndpoint:

    @patch("app.main.predict_for_ticker")
    def test_batch_partial_failure(self, mock_predict):
        def side_effect(ticker):
            if ticker == "FAIL":
                raise FileNotFoundError("No model")
            return {**_MOCK_PREDICTION, "ticker": ticker}

        mock_predict.side_effect = side_effect
        response = client.post("/batch", json={"tickers": ["AAPL", "FAIL"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert "FAIL" in data["errors"]

    def test_empty_tickers_rejected(self):
        response = client.post("/batch", json={"tickers": []})
        assert response.status_code == 422
