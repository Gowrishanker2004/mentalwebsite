from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import jwt
import datetime
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import base64
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from deep_translator import GoogleTranslator

# === CONFIG ===
SECRET_KEY = "your_secret_key"
ALLOWED_EXTENSIONS = {"wav", "mp3"}

# === INIT ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app)
db = SQLAlchemy(app)

# === DATABASE MODEL ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# === LOAD MODELS ===
face_model = load_model("face_emotion_model.h5", compile=False)
voice_model = load_model("voice_emotion_recognition_model.h5", compile=False)

# === UTILS ===
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Corrected MFCC Feature Extraction for 3D input (Conv1D)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Shape: (40,)
    return np.expand_dims(mfcc_mean, axis=0)  # Shape: (1, 40)

def predict_voice_emotion(file_path):
    features = extract_features(file_path)
    predictions = voice_model.predict(features)
    emotion = np.argmax(predictions)
    labels = ['angry', 'happy', 'sad', 'neutral']
    return labels[emotion]

def predict_face_emotion(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    prediction = face_model.predict(image_array)
    emotion = np.argmax(prediction)
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return labels[emotion]

def determine_final_emotion(face, voice):
    if face == voice:
        return face
    if face == "neutral":
        return voice
    if voice == "neutral":
        return face
    priority = ["sad", "angry", "fear", "happy", "surprise", "neutral"]
    return face if priority.index(face) < priority.index(voice) else voice

# === JWT DECORATOR ===
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token is missing!"}), 403
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({"message": "Token is invalid!"}), 403
        return f(*args, **kwargs)
    return decorated

# === AUTH ===
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"message": "Username and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists"}), 400

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully!"}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"message": "Username and password required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "Invalid credentials"}), 401

    token = jwt.encode({
        "user": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({"message": "Login successful", "token": token})

# === FACE EMOTION (CAMERA) ===
@app.route("/api/predict_face_emotion", methods=["POST"])
def predict_face():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        image_bytes = base64.b64decode(image_base64.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = np.expand_dims(img, axis=-1)
        emotion = predict_face_emotion(img)
        return jsonify({"face_emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === VOICE EMOTION ===
@app.route("/api/predict_voice_emotion", methods=["POST"])
def voice_emotion():
    audio = request.files["audio"]
    if audio and allowed_file(audio.filename):
        filename = secure_filename(audio.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio.save(path)
        emotion = predict_voice_emotion(path)
        os.remove(path)
        return jsonify({"voice_emotion": emotion})
    return jsonify({"error": "Invalid file"}), 400

# === FUSION LOGIC ===
@app.route("/api/final_emotion", methods=["POST"])
def fusion():
    face_emotion = request.json.get("face_emotion")
    voice_emotion = request.json.get("voice_emotion")
    final = determine_final_emotion(face_emotion, voice_emotion)
    return jsonify({"final_emotion": final})

# === TRANSLATION (Tamil to English) ===
@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.get_json()
    tamil_text = data.get("text", "")
    english_text = GoogleTranslator(source='ta', target='en').translate(tamil_text)
    return jsonify({"translated": english_text})

# === SIMPLE CHATBOT (Rule-based) ===
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    prompt = data.get("prompt", "").lower()

    if "hello" in prompt or "hi" in prompt:
        response = "Hello! How can I help you today?"
    elif "sad" in prompt:
        response = "I sense sadness. Would you like to try music therapy?"
    elif "happy" in prompt:
        response = "I'm glad to hear that! Keep smiling! 😊"
    elif "anxiety" in prompt:
        response = "Deep breathing might help you feel better."
    else:
        response = "I'm here to listen. Tell me more."

    return jsonify({"response": response})

# === HOME PAGE ===
@app.route("/")
def index():
    return render_template("index.html")

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    with app.app_context():
        db.create_all()
    app.run(debug=True)
