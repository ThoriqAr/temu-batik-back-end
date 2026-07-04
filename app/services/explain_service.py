from app.services.scan_service import (
    scan_image
)

from app.services.gradcam_service import (
    generate_gradcam
)

from app.services.visualization_service import (
    build_visualizations
)

from app.services.explanation_service import (
    generate_explanation
)


def explain_image(
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
            "explanation": None
        }

    inference = scan["inference"]

    gradcam_result = generate_gradcam(
        model=inference["model"],
        processed_image=inference["image_tensor"],
        pred_index=inference["pred_index"],
        backbone_name=inference["model_data"]["backbone"],
        last_conv_layer_name=inference["model_data"]["last_conv_layer"]
    )

    heatmap = gradcam_result["raw_heatmap"]

    visualization_result = build_visualizations(
        inference["original_image"],
        heatmap
    )

    explanation_result = generate_explanation(
        heatmap=heatmap,
        predicted_class=inference["predicted_class"]
    )

    return {
        "success": True,
        "validation": validation,
        "predicted_class": inference["predicted_class"],
        "confidence": inference["confidence"],
        "predictions":inference["predictions"][0].tolist(),
        "focus_region":explanation_result["focus_region"],
        "focus_percentage":explanation_result["focus_percentage"],
        "attention_distribution":explanation_result["attention_distribution"],
        "active_regions":explanation_result["active_regions"],
        "region_scores":explanation_result["region_scores"],
        "explanation":explanation_result["explanation"],
        "overlay_image":visualization_result["overlay_image"]
    }