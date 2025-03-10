from flask import Flask
import logging
import os
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
logger.info(f"Model loaded with {model}")

# segments, info = model.transcribe("audio.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

@app.route("/")
def hello_world():
    try:
        # Check if file exists
        audio_path = "test.wav"
        if not os.path.exists(audio_path):
            return f"<p>Error: File {audio_path} not found</p>"
            
        segments, info = model.transcribe(audio_path, beam_size=5)
        segments_list = list(segments)  # Convert generator to list
        
        result = f"<p>Detected language: {info.language} (probability: {info.language_probability})</p>"
        result += "<ul>"
        for segment in segments_list:
            result += f"<li>[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}</li>"
        result += "</ul>"
        
        return result
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Transcription error: {error_msg}")
        return f"<p>Error during transcription: {str(e)}</p><pre>{error_msg}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)