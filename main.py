from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from openai import OpenAI
import logging
import os
import traceback
import io

system_prompt = """
You are an expert text transcription assistant designed to correct errors in text while preserving
the original meaning and intent. Your sole purpose is to receive text that may contain gramatical
errors and produce a corrected version.

## Context About the Text Creator
- Is a computer science engineer.
- His development environment is macOS/Linux.

## Your Responsibilities:
- Correct spelling errors and grammatical mistakes
- Fix sentence structure maintaining the original meaning
- The author prefers lowercaps to seem more casual
- Maintain technical terminology accuracy based on the context provided

## What *NOT* to Do:
- Do not add new information or expand on ideas
- Do not remove content unless it's clearly redundant
- Do not alter specialized terminology unless incorrectly used
- Do not comment on the quality of the writing
- Do *not* ever try to answer the text's original question. That is not your task.

## Output Format:
Provide *only* the corrected text without explanations, remarks or comments.
"""

hotwords = "js react nextjs tokio async await rust typescript git github vscode cli api json tcp ip ssh"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

logger.info(f"'large-v3' model loaded")

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)


@app.route("/")
def hello_world():
    return "Hello, World!"


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
            hotwords=hotwords,
        )

        result = " ".join(segment.text for segment in segments)
        return jsonify({"text": result})
    except Exception as e:
        print(str(e))
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/clean-transcription", methods=["POST"])
def clean_transcription():
    transcription = request.get_json()
    original_text = transcription["text"]
    try:
        response = openai.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f'Please correct the following text: "{original_text}"',
                },
            ],
        )

        return jsonify(
            {
                "success": True,
                "original_text": original_text,
                "text": response.choices[0].message.content,
            }
        )

    except Exception as e:
        logger.error(f"Clean transcription error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
