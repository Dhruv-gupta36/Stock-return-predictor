import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logging_config import setup_logging

setup_logging(level="INFO", log_dir="logs", log_file="api.log")
logger = logging.getLogger("api")

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from app.inference import (
    get_available_tickers,
    predict_for_ticker,
    preload_all_models,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API startup - preloading models...")
    preload_all_models()
    logger.info("API ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Stock Return Predictor API",
    description="ML-powered next-day stock return prediction using LightGBM.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "detail": "Internal server error."},
    )


@app.get("/", response_model=Dict[str, Any])
def root():
    return {
        "service": "Stock Return Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "batch": "POST /batch",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    available = get_available_tickers()
    return HealthResponse(
        status="ok",
        model_loaded=len(available) > 0,
        available_tickers=available,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(request: PredictRequest):
    ticker = request.ticker
    try:
        result = predict_for_ticker(ticker)
        return PredictResponse(status="success", data=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error for {ticker}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed for {ticker}.",
        )


@app.post("/batch", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    predictions = []
    errors: Dict[str, str] = {}

    for ticker in request.tickers:
        ticker = ticker.upper().strip()
        try:
            result = predict_for_ticker(ticker)
            predictions.append(result)
        except FileNotFoundError:
            errors[ticker] = f"No trained model found for {ticker}."
        except ValueError as exc:
            errors[ticker] = str(exc)
        except Exception as exc:
            logger.error(f"Batch prediction error for {ticker}: {exc}")
            errors[ticker] = "Internal prediction error."

    return BatchPredictResponse(status="success", predictions=predictions, errors=errors)
