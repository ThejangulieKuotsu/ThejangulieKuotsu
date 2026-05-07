from flask import Flask, request, jsonify, render_template

from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
import io
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

## Ensure ffmpeg is correctly set up for pydub
AudioSegment.converter = "ffmpeg"  # Modify this if needed

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    recognizer = sr.Recognizer()

    # Check if an audio file was uploaded
    if 'audio' not in request.files:
        print("❌ No audio file received")
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    print(f"✅ Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")

    try:
        # Read the file into memory
        audio_data = audio_file.read()
        print(f"📦 Audio file size: {len(audio_data)} bytes")

        # Convert WebM to WAV using pydub
        webm_audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        wav_io = io.BytesIO()
        webm_audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Use speech_recognition to transcribe audio
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)

        # Perform speech recognition
        text = recognizer.recognize_google(audio)
        print(f"📝 Transcription: {text}")

        return jsonify({'transcript': text})

    except sr.UnknownValueError:
        print("❌ Speech not recognized")
        return jsonify({'error': 'Speech not recognized'}), 400
    except sr.RequestError:
        print("❌ Google API request failed")
        return jsonify({'error': 'API unavailable'}), 500
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return jsonify({'error': f"Processing error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
