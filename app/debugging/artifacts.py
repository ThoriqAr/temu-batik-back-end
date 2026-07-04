import uuid
import cv2
import numpy as np

from pathlib import Path

from app.core.config import (
    ENABLE_DEBUG_ARTIFACTS,
    UPLOAD_DIR,
    HEATMAP_DIR
)


def save_uploaded_image(
    image_bytes: bytes,
    extension: str = ".jpg"
) -> Path | None:

    if not ENABLE_DEBUG_ARTIFACTS:
        return None

    filename = (
        f"{uuid.uuid4()}"
        f"{extension}"
    )

    destination = (
        UPLOAD_DIR /
        filename
    )

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    cv2.imwrite(
        str(destination),
        image
    )

    return destination


def save_overlay(
    overlay
) -> Path | None:

    if not ENABLE_DEBUG_ARTIFACTS:
        return None

    filename = (
        f"{uuid.uuid4()}.jpg"
    )

    destination = (
        HEATMAP_DIR /
        filename
    )

    cv2.imwrite(
        str(destination),
        overlay
    )

    return destination