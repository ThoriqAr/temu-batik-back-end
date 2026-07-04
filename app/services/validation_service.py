from app.services.embedding_similarity_service import (
    compute_similarity
)


VALID_THRESHOLD = 0.90

UNKNOWN_THRESHOLD = 0.80

MIN_AGREEMENT = 0.80


def build_validation_response(
    *,
    is_valid: bool,
    status: str,
    severity: str,
    title: str,
    message: str,
    similarity_result: dict
):

    return {
        "is_valid": is_valid,
        "status": status,
        "severity": severity,
        "title": title,
        "message": message,
        "metrics": similarity_result["statistics"]
    }


def validate_embedding(
    embedding,
    model_name: str
):

    similarity_result = compute_similarity(
        query_embedding=embedding,
        model_name=model_name,
        top_k=5
    )

    statistics = similarity_result["statistics"]

    max_similarity = statistics["max_similarity"]

    agreement = statistics["agreement"]

    if (

        max_similarity >= VALID_THRESHOLD

        and

        agreement >= MIN_AGREEMENT

    ):

        return build_validation_response(

            is_valid=True,
            status="valid",
            severity="success",
            title="",
            message="",
            similarity_result=similarity_result

        )

    if max_similarity >= UNKNOWN_THRESHOLD:

        return build_validation_response(

            is_valid=False,
            status="unknown_batik",
            severity="info",
            title="Motif belum dikenali",
            message=(
                "Gambar tampak merupakan motif batik, "
                "namun jenis motif tersebut belum tersedia "
                "pada aplikasi saat ini. "
                "Silakan gunakan salah satu motif "
                "yang didukung aplikasi."
            ),
            similarity_result=similarity_result

        )

    return build_validation_response(

        is_valid=False,
        status="non_batik",
        severity="warning",
        title="Motif tidak dapat dikenali",
        message=(
            "Gambar yang dipilih belum dapat dikenali "
            "sebagai motif batik. "
            "Pastikan motif terlihat jelas, "
            "pencahayaan cukup, "
            "dan objek yang difoto merupakan kain batik."
        ),
        similarity_result=similarity_result
        
    )