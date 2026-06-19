# Stock Return Predictor

[![CI](https://github.com/yourusername/stock-return-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/stock-return-predictor/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)

ML-powered next-day stock return prediction using LightGBM, LSTM, Monte Carlo simulation, and SHAP explainability. Deployed via FastAPI and Docker.

---

## Quick Start

```bash
git clone https://github.com/yourusername/stock-return-predictor.git
cd stock-return-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed data/external models reports/figures logs

# Train
python train.py --ticker AAPL --no-optuna --no-lstm

# Predict (CLI)
python predict.py --ticker AAPL

# Run API
uvicorn app.main:app --reload --port 8000
```

---

## Docker

```bash
docker compose build
docker compose up api
docker compose --profile training up trainer
docker compose --profile testing up tests
```

---

## API

```bash
# Health
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Batch prediction
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'

# Docs
open http://localhost:8000/docs
```

---

## Tests

```bash
pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
```

---

## Train Options

```bash
python train.py                          # All tickers, Optuna on, LSTM on
python train.py --ticker AAPL            # Single ticker
python train.py --no-optuna              # Skip Optuna tuning
python train.py --no-lstm                # Skip LSTM
python train.py --no-download            # Use cached data
python train.py --no-shap                # Skip SHAP
```

---

## Stack

Python, LightGBM, PyTorch, FastAPI, Docker, yfinance, SHAP, Optuna, scikit-learn
