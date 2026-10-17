"""Unit tests for the packaged emotion detection module."""
import unittest
from unittest.mock import patch, Mock

from EmotionDetection.emotion_detection import emotion_detector


def mocked_response(emotions, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "emotionPredictions": [{"emotion": emotions}]
    }
    return response


class TestEmotionDetector(unittest.TestCase):
    """Test output formatting and HTTP 400 handling without network calls."""

    def test_joy_is_dominant(self):
        scores = {"anger": 0.01, "disgust": 0.02, "fear": 0.03,
                  "joy": 0.90, "sadness": 0.04}
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=mocked_response(scores)):
            result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(set(result), {
            "anger", "disgust", "fear", "joy", "sadness",
            "dominant_emotion"
        })

    def test_anger_is_dominant(self):
        scores = {"anger": 0.90, "disgust": 0.02, "fear": 0.03,
                  "joy": 0.01, "sadness": 0.04}
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=mocked_response(scores)):
            result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust_is_dominant(self):
        scores = {"anger": 0.02, "disgust": 0.90, "fear": 0.03,
                  "joy": 0.01, "sadness": 0.04}
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=mocked_response(scores)):
            result = emotion_detector("I feel disgusted")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness_is_dominant(self):
        scores = {"anger": 0.02, "disgust": 0.03, "fear": 0.04,
                  "joy": 0.01, "sadness": 0.90}
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=mocked_response(scores)):
            result = emotion_detector("I am so sad")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear_is_dominant(self):
        scores = {"anger": 0.02, "disgust": 0.03, "fear": 0.90,
                  "joy": 0.01, "sadness": 0.04}
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=mocked_response(scores)):
            result = emotion_detector("I am really afraid")
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_blank_input(self):
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])
        self.assertIsNone(result["anger"])

    def test_http_400(self):
        response = Mock(status_code=400)
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=response):
            result = emotion_detector("bad request")
        self.assertIsNone(result["dominant_emotion"])
        self.assertEqual(set(result), {
            "anger", "disgust", "fear", "joy", "sadness",
            "dominant_emotion"
        })


if __name__ == "__main__":
    unittest.main()
