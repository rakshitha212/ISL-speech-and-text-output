from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sqlite3
import speech_recognition as sr
import os
import json
import threading
import time
import torch

app = Flask(__name__)

# =========================
# Load LSTM Model (UPDATED)
# =========================
CUSTOM_MODEL_PATH = "model/gesture_model_lstm.h5"
LABELS_PATH = "model/labels.json"

custom_model = None
custom_labels = []

if os.path.exists(CUSTOM_MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
        with open(LABELS_PATH, "r") as f:
            custom_labels = json.load(f)
        print(f"[INFO] LSTM Model loaded: {len(custom_labels)} classes")
    except Exception as e:
        print(f"[WARN] LSTM Model load failed: {e}")

# =========================
# Load INCLUDE Transformer Model (for Gesture-to-Speech)
# =========================
INCLUDE_MODEL_PATH = "model/include50_transformer_small.pth"
INCLUDE_LABELS_PATH = "model/include_labels.json"

include_model = None
include_labels = {}
include_idx_to_label = {}

if os.path.exists(INCLUDE_MODEL_PATH) and os.path.exists(INCLUDE_LABELS_PATH):
    try:
        from model.include_models import Transformer, TransformerConfig
        config = TransformerConfig(size="small")
        include_model = Transformer(config=config, n_classes=50)
        ckpt = torch.load(INCLUDE_MODEL_PATH, map_location="cpu", weights_only=False)
        include_model.load_state_dict(ckpt["model"])
        include_model.eval()
        with open(INCLUDE_LABELS_PATH, "r") as f:
            include_labels = json.load(f)
        include_idx_to_label = {v: k for k, v in include_labels.items()}
        print(f"[INFO] INCLUDE Transformer Model loaded: {len(include_labels)} classes")
    except Exception as e:
        print(f"[WARN] INCLUDE Transformer Model load failed: {e}")

# =========================
# Global Shared State
# =========================
latest_frame = None
locked_word = ""
current_status = "WAITING"
detection_active = False
camera_thread = None
frame_lock = threading.Lock()

# =========================
# Camera Thread
# =========================
def camera_loop():
    global latest_frame, locked_word, current_status, detection_active

    mp_hol = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    # Buffer 2 seconds of frames (60 at 30fps) — INCLUDE model supports up to 256 positions
    BUFFER_LEN = 60
    INFER_EVERY = 5   # Run inference every 5 frames (~6 inferences/sec, reduces noise)
    seq = []
    last_word = ""
    stable_count = 0
    cooldown_until = 0
    infer_count = 0
    # Hand interpolation: carry forward last known position when hand briefly disappears
    last_left_hand = None
    last_right_hand = None
    left_missing_frames = 0
    right_missing_frames = 0
    MAX_MISSING_FRAMES = 10

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    holistic = mp_hol.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while detection_active:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # DRAW LANDMARKS
        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_hol.POSE_CONNECTIONS)
        if res.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS)
        if res.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS)

        feats = []

        # INCLUDE training pipeline: raw x,y scaled to pixel coordinates (1920×1080)
        # No per-frame normalization — model learned on pixel-scale absolute positions
        FRAME_W, FRAME_H = 1920.0, 1080.0

        # Upper body Pose (landmarks 0-24) → 25 keypoints × 2 coords = 50 features
        if res.pose_landmarks:
            for i in range(25):
                lm = res.pose_landmarks.landmark[i]
                feats.extend([lm.x * FRAME_W, lm.y * FRAME_H])
        else:
            feats.extend([0.0] * 50)

        # Left hand (hand1 in INCLUDE convention) → 21 keypoints × 2 coords = 42 features
        # Interpolate missing hands: carry forward last position (INCLUDE interpolates NaN)
        if res.left_hand_landmarks:
            lh_feats = []
            for lm in res.left_hand_landmarks.landmark:
                lh_feats.extend([lm.x * FRAME_W, lm.y * FRAME_H])
            feats.extend(lh_feats)
            last_left_hand = lh_feats[:]
            left_missing_frames = 0
        elif last_left_hand is not None and left_missing_frames < MAX_MISSING_FRAMES:
            feats.extend(last_left_hand)
            left_missing_frames += 1
        else:
            feats.extend([0.0] * 42)
            last_left_hand = None
            left_missing_frames = 0

        # Right hand (hand2 in INCLUDE convention) → 21 keypoints × 2 coords = 42 features
        if res.right_hand_landmarks:
            rh_feats = []
            for lm in res.right_hand_landmarks.landmark:
                rh_feats.extend([lm.x * FRAME_W, lm.y * FRAME_H])
            feats.extend(rh_feats)
            last_right_hand = rh_feats[:]
            right_missing_frames = 0
        elif last_right_hand is not None and right_missing_frames < MAX_MISSING_FRAMES:
            feats.extend(last_right_hand)
            right_missing_frames += 1
        else:
            feats.extend([0.0] * 42)
            last_right_hand = None
            right_missing_frames = 0
        # Total: 50 + 84 = 134 features

        # Add to sequence buffer
        seq.append(feats)
        if len(seq) > BUFFER_LEN:
            seq.pop(0)

        # =========================
        # GESTURE PREDICTION
        # =========================
        status_msg = "READY"
        status_color = (0, 255, 0)
        
        if not res.left_hand_landmarks and not res.right_hand_landmarks:
            status_msg = "NO HANDS DETECTED"
            status_color = (0, 0, 255)
            current_status = "No Hands Detected"
        elif not res.pose_landmarks:
            status_msg = "BODY NOT DETECTED"
            status_color = (0, 0, 255)
            current_status = "Body Not Detected"
        else:
            current_status = "Analyzing..."
            status_msg = "ANALYZING"
            status_color = (255, 255, 0)
            
            if include_model is not None:
                if len(seq) < BUFFER_LEN:
                    current_status = f"Buffering... {len(seq)}/{BUFFER_LEN}"
                else:
                    infer_count += 1
                    # Only run inference every INFER_EVERY frames to reduce noise & CPU
                    if infer_count % INFER_EVERY == 0:
                        # Feed BUFFER_LEN frames directly — model supports up to 256 positions
                        # NO sequence-level normalization — INCLUDE model trained on raw pixel coords
                        sequence = np.array(seq[-BUFFER_LEN:])
                        sequence = sequence.reshape(1, BUFFER_LEN, 134)

                        # PyTorch inference with INCLUDE Transformer
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(sequence)
                            pred = include_model(input_tensor)
                            probs = torch.softmax(pred, dim=-1)
                            conf = float(torch.max(probs))
                            word_idx = int(torch.argmax(probs, dim=-1))
                            word = include_idx_to_label.get(word_idx, "?")

                        # Debug logging (every inference run)
                        hands_status = f"L={'Y' if res.left_hand_landmarks else 'N'} R={'Y' if res.right_hand_landmarks else 'N'}"
                        top3 = torch.topk(probs[0], 3)
                        top3_str = ", ".join(
                            f"{include_idx_to_label[int(top3.indices[i])]}={float(top3.values[i]):.2f}"
                            for i in range(3)
                        )
                        print(f"[DEBUG] pred={word} conf={conf:.3f} | {hands_status} | top3: {top3_str}")

                        # Stability & Output Logic
                        if conf > 0.40:
                            if word == last_word:
                                stable_count += 1
                            else:
                                stable_count = 0
                                last_word = word
                            
                            if stable_count >= 3:
                                if time.time() > cooldown_until:
                                    locked_word = word
                                    current_status = "LOCKED"
                                    cooldown_until = time.time() + 2.0
                                else:
                                    current_status = "Cooldown..."
                            else:
                                current_status = f"Analyzing... {word} ({int(conf*100)}%)"
                        elif conf > 0.10:
                            current_status = f"Analyzing... ({int(conf*100)}%)"
                    # else: keep previous current_status (no update between inference runs)
            elif custom_model is not None:
                # Fallback to old LSTM model if INCLUDE model not loaded
                LSTM_SEQ_LEN = 30
                if len(seq) < LSTM_SEQ_LEN:
                    current_status = f"Buffering... {len(seq)}/{LSTM_SEQ_LEN}"
                else:
                    sequence = np.array(seq[-LSTM_SEQ_LEN:])
                    sequence = sequence - sequence[0]
                    sequence = sequence / (np.max(np.abs(sequence)) + 1e-6)
                    # Pad features from 134 to 150 for old LSTM
                    if sequence.shape[1] == 134:
                        pad_feats = np.zeros((sequence.shape[0], 16))
                        sequence = np.hstack((sequence, pad_feats))
                    sequence = sequence.reshape(1, 30, 150)

                    pred = custom_model.predict(sequence, verbose=0)[0]
                    conf = float(np.max(pred))
                    word = custom_labels[int(np.argmax(pred))]

                    if conf > 0.85:
                        if word == last_word:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_word = word
                        if stable_count >= 4:
                            if time.time() > cooldown_until:
                                locked_word = word
                                current_status = "LOCKED"
                                cooldown_until = time.time() + 2.0
                            else:
                                current_status = "Cooldown..."
                        else:
                            current_status = f"Analyzing... {word} ({int(conf*100)}%)"
                    elif conf > 0.50:
                        current_status = f"Analyzing... ({int(conf*100)}%)"

        if time.time() > cooldown_until and current_status != "LOCKED":
             locked_word = "" # Clear the locked word after cooldown if we are analyzing again

        # Draw status
        cv2.putText(frame, status_msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Display word
        if locked_word:
            cv2.putText(frame, locked_word.upper(), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Encode frame
        _, buf = cv2.imencode('.jpg', frame)
        with frame_lock:
            latest_frame = buf.tobytes()

    cap.release()
    print("[INFO] Camera thread stopped.")

# Camera thread is no longer started globally on init

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/gesture")
def gesture():
    return render_template("gesture.html")

@app.route("/speech")
def speech():
    return render_template("speech.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/get_gesture")
def get_gesture():
    return jsonify({
        "word": locked_word,
        "status": current_status
    })

@app.route("/current_prediction")
def current_prediction():
    return jsonify({"prediction": locked_word})

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    global detection_active, camera_thread, locked_word, current_status
    data = request.get_json(silent=True) or {}
    
    # Allow passing explicit state, else toggle
    if "active" in data:
        new_state = bool(data["active"])
    else:
        new_state = not detection_active

    if new_state and not detection_active:
        detection_active = True
        locked_word = ""
        current_status = "Starting camera..."
        if camera_thread is None or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=camera_loop, daemon=True)
            camera_thread.start()
    elif not new_state and detection_active:
        detection_active = False
        locked_word = ""
        current_status = "WAITING"
        
    return jsonify({"active": detection_active})

@app.route("/detection_status")
def detection_status():
    return jsonify({"active": detection_active})

@app.route("/supported_words")
def supported_words():
    words = sorted(include_labels.keys()) if include_labels else []
    return jsonify({"words": words, "count": len(words)})

# =========================
# Speech to Sign
# =========================
def get_db():
    return sqlite3.connect("database/isl.db")

@app.route("/speech_to_sign", methods=["POST"])
def speech_to_sign():
    rec = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source, duration=1)
            audio = rec.listen(source, timeout=5)

        text = rec.recognize_google(audio)
        word = text.lower()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT gesture_file FROM gestures WHERE word=?", (word,))
        row = cur.fetchone()
        conn.close()

        if row:
            return jsonify({"text": text, "gesture": f"/static/gestures/{row[0]}"})

        gestures = [f"/static/gestures/{c}.gif" for c in word if c.isalpha()]
        return jsonify({"text": text, "gestures": gestures})

    except:
        return jsonify({"text": "Error", "gesture": "/static/gestures/not_found.gif"})


if __name__ == "__main__":
    app.run(debug=False)