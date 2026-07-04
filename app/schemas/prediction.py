from app.schemas.base import (
    BaseApiResponse
)


class PredictionResponse(
    BaseApiResponse
):

    predicted_class: str | None = None

    confidence: float | None = None

    predictions: list[float] | None = None