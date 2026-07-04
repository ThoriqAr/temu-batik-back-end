import numpy as np

from app.core.constants import CLASS_NAMES

from app.core.model_registry import (
    get_model
)

from app.core.embedding_extractor_registry import (
    get_embedding_extractor
)

from app.utils.image import (
    read_image,
    resize_image,
    prepare_image_tensor
)

from app.utils.preprocessing import (
    preprocess_image
)


def prepare_input(
    image_bytes: bytes,
    model_name: str
):

    image = read_image(image_bytes)

    original_image = resize_image(image)

    image_tensor = prepare_image_tensor(
        original_image
    )

    image_tensor = preprocess_image(
        image_tensor,
        model_name
    )

    return original_image, image_tensor


def extract_embedding(
    image_tensor,
    model_name: str
):

    extractor = get_embedding_extractor(model_name)

    if extractor is None:

        raise ValueError(
            f"Embedding extractor "
            f"'{model_name}' not found."
        )

    embedding = extractor.predict(
        image_tensor,
        verbose=0
    )[0]

    return embedding.astype(np.float32)


def predict_tensor(
    image_tensor,
    model_name: str
):

    model_data = get_model(model_name)

    if not model_data:
        raise ValueError(
            f"Model '{model_name}' not found."
        )

    model = model_data["model"]

    predictions = model.predict(
        image_tensor,
        verbose=0
    )

    pred_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][pred_index]
    )

    predicted_class = CLASS_NAMES[
        pred_index
    ]

    return {

        "model": model,

        "model_data": model_data,

        "predictions": predictions,

        "pred_index": pred_index,

        "predicted_class": predicted_class,

        "confidence": confidence

    }


def run_inference(
    image_bytes: bytes,
    model_name: str
):

    original_image, image_tensor = prepare_input(
        image_bytes=image_bytes,
        model_name=model_name
    )

    embedding = extract_embedding(
        image_tensor=image_tensor,
        model_name=model_name
    )

    prediction = predict_tensor(
        image_tensor=image_tensor,
        model_name=model_name
    )

    prediction["original_image"] = original_image

    prediction["image_tensor"] = image_tensor

    prediction["embedding"] = embedding

    return prediction