from fastapi import APIRouter
from app.core.model_registry import get_model
from app.core.model_registry import MODEL_REGISTRY
from app.core.embedding_extractor_registry import EMBEDDING_REGISTRY
from app.core.embedding_database_registry import EMBEDDING_DATABASES

router = APIRouter()


@router.get("/models")
def get_models():

    return {
        "loaded_models": list(MODEL_REGISTRY.keys())
    }

@router.get("/models/{model_name}/layers")
def get_model_layers(model_name: str):

    model_data = get_model(model_name)

    if not model_data:
        return {
            "error": "Model not found"
        }

    model = model_data["model"]

    layer_names = []

    for layer in model.layers:
        layer_names.append(layer.name)

    return {
        "layers": layer_names
    }

@router.get("/embeddings")
def get_embedding_models():

    return {
        "embedding_extractors":
            list(
                EMBEDDING_REGISTRY.keys()
            )
    }

@router.get("/embedding-databases")
def embedding_databases():

    return {
        model_name: {
            "num_embeddings":
                len(data["embeddings"]),

            "dimension":
                int(
                    data["embeddings"].shape[1]
                ),

            "num_classes":
                data["metadata"]["num_classes"]
        }

        for model_name, data
        in EMBEDDING_DATABASES.items()

    }