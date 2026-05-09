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

    # Real-time buffer (3 seconds at 30fps)
    BUFFER_LEN = 90 
    seq = []
    last_word = ""
    stable_count = 0
    cooldown_until = 0

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

        # Add to sequence buffer
        seq.append(feats)
        if len(seq) > BUFFER_LEN:
            seq.pop(0)

        # =========================
        # LSTM PREDICTION
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
            
            if custom_model is not None:
                if len(seq) < BUFFER_LEN:
                    current_status = f"Buffering... {len(seq)}/{BUFFER_LEN}"
                else:
                    # Check if we have enough movement to bother predicting
                    recent_seq = np.array(seq[-30:])
                    variance = np.var(recent_seq)
                    
                    # If variance is very low, hand is completely still, might be noise
                    if variance > 0.0001:
                        # Sub-sample 30 frames from the 90-frame buffer (every 3rd frame)
                        sample_seq = seq[::3] 
                        sequence = np.array(sample_seq)

                        # NORMALIZATION
                        sequence = sequence - sequence[0]
                        sequence = sequence / (np.max(np.abs(sequence)) + 1e-6)
                        sequence = sequence.reshape(1, 30, 150)

                        pred = custom_model.predict(sequence, verbose=0)[0]
                        conf = float(np.max(pred))
                        word = custom_labels[int(np.argmax(pred))]

                        # Stability & Output Logic
                        if conf > 0.85: # Increased threshold
                            if word == last_word:
                                stable_count += 1
                            else:
                                stable_count = 0
                                last_word = word
                            
                            if stable_count >= 4: # Require more stable frames
                                if time.time() > cooldown_until:
                                    locked_word = word
                                    current_status = "LOCKED"
                                    cooldown_until = time.time() + 2.0 # 2s cooldown
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