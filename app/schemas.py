from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, example="AAPL")
    model_subdir: Optional[str] = Field(default=None)

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_alpha(cls, v: str) -> str:
        v = v.upper().strip()
        if not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid ticker symbol: {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        }
    }


class PredictionData(BaseModel):
    ticker: str
    prediction_date: str
    predicted_return: float
    predicted_return_pct: str
    predicted_direction: str
    confidence: float
    current_price: float


class PredictResponse(BaseModel):
    status: str = Field(default="success")
    data: PredictionData


class ErrorResponse(BaseModel):
    status: str = Field(default="error")
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_tickers: list
    version: str = "1.0.0"


class BatchPredictRequest(BaseModel):
    tickers: list = Field(..., min_length=1, max_length=20, example=["AAPL", "MSFT"])


class BatchPredictResponse(BaseModel):
    status: str = Field(default="success")
    predictions: list
    errors: dict
