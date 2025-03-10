from flask import Flask
import logging
import os
import traceback
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
logger.info(f"Model loaded: {model.model}")

@app.route("/")
def hello_world():
    try:
        audio_path = "test.wav"
        if not os.path.exists(audio_path):
            return f"Error: File {audio_path} not found"
            
        start_time = time.time()

        segments, _ = model.transcribe(audio_path, beam_size=5, language="en", without_timestamps=True)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)