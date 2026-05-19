from fastapi import APIRouter
from app.core.model_registry import get_model
from app.core.model_registry import MODEL_REGISTRY

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