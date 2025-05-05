from flask import Flask, request, jsonify
import os
import tempfile
# import torch
import numpy as np
import librosa
import subprocess
import io
from pydub import AudioSegment
# from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Global variables
MODEL_PATH = "speech_emotion_recognition_model"  # Update this with your saved model path
SAMPLING_RATE = 16000
MAX_LENGTH = 32000

# Load model and tokenizer
kado_model = load_model("model/text_classifier_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    kado_tokenizer = pickle.load(f)

maxlen = 79  # Must match the value used during training


@app.route('/text', methods=['POST'])
def predictText():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sequence = kado_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')

    prediction = kado_model.predict(padded)
    predicted_class = int(prediction.argmax(axis=1)[0])

    print("Tokenized:", sequence)
    print("Padded:", padded)
    print("Prediction:", prediction)

    return jsonify({
        "text": text,
        "predicted_class": predicted_class,
        "confidence": float(prediction[0][predicted_class]),
    })


# Load the model and processor
def load_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        # Create label mapping (you'll need to adjust this based on your actual labels)
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
        label_map = {idx: label for idx, label in enumerate(emotions)}
        return processor, model, label_map
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        return None, None, None

processor, model, label_map = load_model()

def preprocess_audio(audio_file, sr=SAMPLING_RATE):
    try:
        app.logger.info(f"Processing audio file: {audio_file}")
        
        # First try to convert the file to proper WAV format using pydub
        try:
            # Detect format and convert to WAV
            audio = AudioSegment.from_file(audio_file)
            
            # Export as proper WAV with correct parameters
            wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_file.close()
            
            audio = audio.set_frame_rate(sr)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_sample_width(2)  # 16-bit
            audio.export(wav_file.name, format="wav")
            
            app.logger.info(f"Successfully converted audio to standard WAV: {wav_file.name}")
            
            # Load the converted WAV file using librosa
            speech, _ = librosa.load(wav_file.name, sr=sr)
            
            # Clean up temporary file
            os.unlink(wav_file.name)
            
        except Exception as convert_err:
            app.logger.error(f"Error converting audio: {str(convert_err)}")
            
            # Fallback: try with librosa directly
            try:
                speech, _ = librosa.load(audio_file, sr=sr)
                app.logger.info("Successfully loaded audio with librosa")
            except Exception as librosa_err:
                app.logger.error(f"Error loading with librosa: {str(librosa_err)}")
                raise
        
        # Process audio data
        if len(speech) > MAX_LENGTH:
            speech = speech[:MAX_LENGTH]
        else:
            speech = np.pad(speech, (0, MAX_LENGTH - len(speech)), 'constant')
        
        # Process with wav2vec processor
        inputs = processor(speech, sampling_rate=sr, return_tensors='pt', padding=True)
        input_values = inputs.input_values.squeeze()
        
        return input_values
    
    except Exception as e:
        app.logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
        raise

# Predict emotion function
def predict_emotion(audio_tensor):
    with torch.no_grad():
        outputs = model(audio_tensor.unsqueeze(0))
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()
        
        # Get predicted emotion label
        predicted_emotion = label_map[predicted_class]
        
        # Get confidence scores for all emotions
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        confidence_scores = {label_map[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return predicted_emotion, confidence_scores

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        app.logger.info(f"Received file: {file.filename}")
        app.logger.info(f"Content type: {file.content_type}")
        app.logger.info(f"Headers: {dict(request.headers)}")
        app.logger.info(f"Processing file: {file.filename}")
        
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()
        
        app.logger.info("File saved, preprocessing audio...")
        # Preprocess the audio
        audio_tensor = preprocess_audio(temp_file.name)
        
        app.logger.info("Running prediction...")
        # Get prediction
        emotion, confidence_scores = predict_emotion(audio_tensor)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        app.logger.info(f"Prediction successful: {emotion}")
        # Return prediction
        return jsonify({
            'predicted_emotion': emotion,
            'confidence_scores': confidence_scores
        })
        
    except Exception as e:
        app.logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        # Clean up temporary file if it exists
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        return jsonify({
            'error': str(e),
            'message': 'The audio format is not supported. Please ensure you are sending a standard WAV file (16kHz, mono, 16-bit).'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    if processor is None or model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'ok', 'message': 'Service is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)