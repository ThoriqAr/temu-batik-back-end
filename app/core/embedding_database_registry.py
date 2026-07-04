import json
import numpy as np

from app.core.config import EMBEDDING_DIR


EMBEDDING_DATABASES = {}


def load_embedding_databases():

    EMBEDDING_DATABASES.clear()

    for embedding_path in EMBEDDING_DIR.iterdir():

        if not embedding_path.is_dir():
            continue

        model_name = embedding_path.name

        embeddings = np.load(
            embedding_path / "embeddings.npy"
        )

        labels = np.load(
            embedding_path / "labels.npy"
        )

        with open(
            embedding_path / "metadata.json",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        EMBEDDING_DATABASES[model_name] = {
            "embeddings": embeddings,
            "labels": labels,
            "metadata": metadata
        }

        print(
            f"Embedding database loaded: "
            f"{model_name} "
            f"({len(embeddings)} vectors)"
        )


def get_embedding_database(
    model_name: str
):

    return EMBEDDING_DATABASES.get(model_name)