from pydantic import BaseModel


class ValidationMetrics(BaseModel):

    max_similarity: float

    mean_similarity: float

    mean_topk_similarity: float

    std_topk_similarity: float

    margin: float

    agreement: float


class ValidationResponse(BaseModel):

    is_valid: bool

    status: str

    severity: str

    title: str

    message: str

    metrics: ValidationMetrics


class BaseApiResponse(BaseModel):

    success: bool

    validation: ValidationResponse