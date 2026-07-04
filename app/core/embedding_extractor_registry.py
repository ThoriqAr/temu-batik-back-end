from tensorflow.keras.models import Model

from app.core.model_registry import MODEL_REGISTRY
from app.core.constants import MODEL_CONFIGS


EMBEDDING_REGISTRY = {}


def load_embedding_extractors():

    EMBEDDING_REGISTRY.clear()

    for model_name, model_data in MODEL_REGISTRY.items():

        model = model_data["model"]

        config = MODEL_CONFIGS[model_name]

        embedding_layer_name = config["embedding_layer"]

        extractor = Model(
            inputs=model.input,
            outputs=model.get_layer(
                embedding_layer_name
            ).output,
            name=f"{model_name}_embedding_extractor"
        )

        EMBEDDING_REGISTRY[model_name] = extractor

        print(
            f"Embedding extractor loaded: {model_name}"
        )


def get_embedding_extractor(model_name: str):

    return EMBEDDING_REGISTRY.get(model_name)