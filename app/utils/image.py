import cv2
import numpy as np

from app.core.constants import IMAGE_SIZE


def read_image(image_bytes: bytes):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Invalid image."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def resize_image(image: np.ndarray):

    image = cv2.resize(image, IMAGE_SIZE)

    return image


def prepare_image_tensor(image: np.ndarray):

    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)

    return image