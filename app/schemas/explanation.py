from app.schemas.base import (
    BaseApiResponse
)


class ExplanationResponse(
    BaseApiResponse
):

    predicted_class: str | None = None

    confidence: float | None = None

    focus_region: str | None = None

    focus_percentage: float | None = None

    peak_activation: float | None = None

    attention_distribution: str | None = None

    active_regions: int | None = None

    region_scores: dict[str, float] | None = None

    explanation: str | None = None

    overlay_image: str | None = None