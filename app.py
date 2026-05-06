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
        print(f"[INFO] Model loaded: {len(custom_labels)} classes")
    except Exception as e:
        print(f"[WARN] Model load failed: {e}")

# =========================
# Global Shared State
# =========================
latest_frame = None
current_word = ""
detection_active = True
frame_lock = threading.Lock()

# =========================
# Camera Thread
# =========================
def camera_loop():
    global latest_frame, current_word

    mp_hol = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    SEQ_LEN = 30
    seq = []
    last_word = ""
    stable_count = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    holistic = mp_hol.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        feats = []

        # Pose (66)
        if res.pose_landmarks:
            lh = res.pose_landmarks.landmark[23]
            rh = res.pose_landmarks.landmark[24]
            hx, hy = (lh.x + rh.x) / 2, (lh.y + rh.y) / 2
            ls = res.pose_landmarks.landmark[11]
            rs = res.pose_landmarks.landmark[12]
            sd = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) or 1

            for lm in res.pose_landmarks.landmark:
                feats.extend([(lm.x - hx) / sd, (lm.y - hy) / sd])
        else:
            feats.extend([0.0] * 66)

        # Hands (84)
        for hand in [res.left_hand_landmarks, res.right_hand_landmarks]:
            if hand:
                wr = hand.landmark[0]
                md = max(np.sqrt((lm.x - wr.x) ** 2 + (lm.y - wr.y) ** 2)
                         for lm in hand.landmark) or 1
                for lm in hand.landmark:
                    feats.extend([(lm.x - wr.x) / md, (lm.y - wr.y) / md])
            else:
                feats.extend([0.0] * 42)

        # Add to sequence
        seq.append(feats)
        if len(seq) > SEQ_LEN:
            seq.pop(0)

        # =========================
        # LSTM PREDICTION (FIXED)
        # =========================
        if detection_active and custom_model is not None:
            if len(seq) == SEQ_LEN:

                sequence = np.array(seq)

                # NORMALIZATION (Relative to first frame)
                sequence = sequence - sequence[0]
                sequence = sequence / (np.max(np.abs(sequence)) + 1e-6)

                sequence = sequence.reshape(1, SEQ_LEN, 150)

                pred = custom_model.predict(sequence, verbose=0)[0]
                conf = float(np.max(pred))
                word = custom_labels[int(np.argmax(pred))]

                # Stability logic
                if word == last_word:
                    stable_count += 1
                else:
                    stable_count = 0

                last_word = word

                if stable_count > 5 and conf > 0.70:
                    current_word = word
                else:
                    current_word = ""

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Display word
        if current_word:
            cv2.putText(frame, current_word.upper(), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Encode frame
        _, buf = cv2.imencode('.jpg', frame)
        with frame_lock:
            latest_frame = buf.tobytes()

    cap.release()


# Start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

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
    return jsonify({"word": current_word})

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
    global detection_active
    detection_active = not detection_active
    return jsonify({"active": detection_active})

@app.route("/detection_status")
def detection_status():
    return jsonify({"active": detection_active})

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