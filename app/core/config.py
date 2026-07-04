from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

TEMP_DIR = BASE_DIR / "temp"
UPLOAD_DIR = TEMP_DIR / "uploads"
HEATMAP_DIR = TEMP_DIR / "heatmaps"

EMBEDDING_DIR = (
    BASE_DIR /
    "artifacts" /
    "embeddings"
)

def get_bool(name: str, default=False):

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in (
        "true",
        "1",
        "yes",
        "on"
    )


ENABLE_DEBUG_ARTIFACTS = get_bool(
    "ENABLE_DEBUG_ARTIFACTS",
    default=False
)

if ENABLE_DEBUG_ARTIFACTS:

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HEATMAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )