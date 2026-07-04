from app.services.inference_service import (
    run_inference
)

from app.services.validation_service import (
    validate_embedding
)

from app.debugging.artifacts import (
    save_uploaded_image
)


def scan_image(
    image_bytes: bytes,
    model_name: str
):

    save_uploaded_image(
        image_bytes
    )

    inference = run_inference(
        image_bytes=image_bytes,
        model_name=model_name
    )

    validation = validate_embedding(
        embedding=inference["embedding"],
        model_name=model_name
    )

    return {
        "inference": inference,
        "validation": validation
    }