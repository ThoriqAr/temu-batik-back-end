from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.schemas.explanation import ExplanationResponse

from app.services.explain_service import (
    explain_image
)

router = APIRouter()


@router.post(
    "/explain/{model_name}",
    response_model=ExplanationResponse
)
async def explain(

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

    try:

        return explain_image(
            image_bytes=image_bytes,
            model_name=model_name
        )

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )