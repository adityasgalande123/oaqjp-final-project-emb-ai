"""Flask web interface for Watson NLP emotion detection."""
from flask import Flask, jsonify, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the web interface."""
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET", "POST"])
def sent_analyzer():
    """Validate submitted text and return emotion analysis."""
    if request.method == "POST":
        text_to_analyze = request.form.get("textToAnalyze", "")
    else:
        text_to_analyze = request.args.get("textToAnalyze", "")

    if not text_to_analyze.strip():
        return jsonify({"error": "Please enter some text to analyze."}), 400

    try:
        result = emotion_detector(text_to_analyze)
    except (KeyError, ValueError, TypeError):
        app.logger.exception("Unexpected response from emotion service")
        return jsonify({"error": "Unable to analyze this text. Please try again."}), 502
    except Exception:
        app.logger.exception("Emotion service request failed")
        return jsonify({"error": "Emotion service is currently unavailable."}), 502

    if result.get("dominant_emotion") is None:
        return jsonify({"error": "Invalid text! Please try again."}), 400

    return jsonify({
        "text": text_to_analyze,
        "emotions": result,
        "dominant_emotion": result["dominant_emotion"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
