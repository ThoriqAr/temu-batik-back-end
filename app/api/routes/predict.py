from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import predict_image

router = APIRouter()


@router.post(
    "/predict/{model_name}",
    response_model=PredictionResponse
)
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):

    if model_name not in [
      "mobilenetv2",
      "resnet50"
    ]:
      raise HTTPException(
          status_code=400,
          detail="Invalid model name"
      )

    image_bytes = await file.read()

    result = predict_image(
        image_bytes=image_bytes,
        model_name=model_name
    )

    return result