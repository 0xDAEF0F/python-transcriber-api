from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import logging
import os
import traceback
import time
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

logger.info(f"'large-v3' model loaded")


@app.route("/")
def hello_world():
    try:
        audio_path = "test.wav"
        if not os.path.exists(audio_path):
            return f"Error: File {audio_path} not found"

        start_time = time.time()

        segments, _ = model.transcribe(
            audio_path, beam_size=5, language="en", without_timestamps=True
        )

        end_time = time.time()

        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        result = f""
        for segment in segments:
            result += f"{segment.text} "
        result += "\n"

        return result
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Transcription error: {error_msg}")
        return f"Error during transcription: {str(e)}"


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_data = request.get_data()

    if len(audio_data) == 0:
        return jsonify({"error": "No audio data provided"}), 400

    try:
        audio_file = io.BytesIO(audio_data)

        segments, _ = model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            without_timestamps=True,
            task="transcribe",
        )

        result = " ".join(segment.text for segment in segments)
        return jsonify({"text": result + "\n"})
    except Exception as e:
        print(str(e))
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
