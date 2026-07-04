from app.services.scan_service import (
    scan_image
)


def predict_image(
    image_bytes: bytes,
    model_name: str
):

    scan = scan_image(
        image_bytes=image_bytes,
        model_name=model_name
    )

    validation = scan["validation"]

    if not validation["is_valid"]:
        
        return {
            "success": False,
            "validation": validation,
            "prediction": None
        }

    inference = scan["inference"]

    return {
        "success": True,
        "validation": validation,
        "predicted_class":inference["predicted_class"],
        "confidence":inference["confidence"],
        "predictions":inference["predictions"][0].tolist()
    }