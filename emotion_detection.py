"""Watson NLP emotion detection with normalized output."""
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
}
EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the dominant emotion.

    A service HTTP 400 response returns None values in the expected
    dictionary shape. Other HTTP errors are raised for the caller.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {**dict.fromkeys(EMOTION_KEYS), "dominant_emotion": None}

    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        URL, headers=HEADERS, json=payload, timeout=30
    )

    if response.status_code == 400:
        return {**dict.fromkeys(EMOTION_KEYS), "dominant_emotion": None}

    response.raise_for_status()
    data = response.json()
    emotions = data["emotionPredictions"][0]["emotion"]
    result = {key: emotions.get(key) for key in EMOTION_KEYS}

    available = {key: value for key, value in result.items()
                 if isinstance(value, (int, float))}
    result["dominant_emotion"] = (
        max(available, key=available.get) if available else None
    )
    return result
