# Final Project

Emotion Detection application using the Watson NLP EmotionPredict service and Flask.

## Project structure
- `emotion_detection.py`: initial/root module for the early activities.
- `EmotionDetection/emotion_detection.py`: packaged module for later activities.
- `EmotionDetection/__init__.py`: package initializer.
- `test_emotion_detection.py`: unit tests.
- `server.py`: Flask web application.
- `templates/index.html`: browser interface.

## Setup
```bash
python3 -m pip install -r requirements.txt
python3 server.py
```
Open the forwarded port for Flask (5000) in your lab environment.

The Watson NLP service must be reachable from the environment for live predictions.
