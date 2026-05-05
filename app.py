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
# Load Custom ML Model
# =========================
CUSTOM_MODEL_PATH = "model/gesture_model_custom.h5"
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
detection_active = True  # Can be toggled via /toggle_detection
frame_lock = threading.Lock()

# =========================
# Background Camera Thread
# =========================
def camera_loop():
    global latest_frame, current_word

    mp_hol = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    SEQ_LEN = 30
    seq = []

    # Try DirectShow first (fixes Windows lock issues), fallback to default
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] DirectShow failed, trying default...")
        cap = cv2.VideoCapture(0)

    # Retry loop for camera
    retry_count = 0
    while not cap.isOpened() and retry_count < 10:
        print(f"[WARN] Camera locked, retrying... ({retry_count}/10)")
        time.sleep(1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        retry_count += 1

    if not cap.isOpened():
        print("[ERROR] Cannot open camera after retries")
        # Generate a blank frame with an error message so UI doesn't hang
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "CAMERA ERROR - PLEASE RESTART APP", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', blank_frame)
        with frame_lock:
            latest_frame = buf.tobytes()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Camera started successfully")

    holistic = mp_hol.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )

    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        fc += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert to RGB and run MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # --- Draw Face Mesh ---
        if res.face_landmarks:
            mp_draw.draw_landmarks(
                frame, res.face_landmarks, mp_hol.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(0, 200, 150), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=1)
            )

        # --- Draw Pose ---
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2)
            )

        # --- Draw Hands ---
        if res.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS)
        if res.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS)

        # --- Feature Extraction every frame for faster filling ---
        feats = []
        tracking_status = "Tracking..."

        # Pose (66)
        if res.pose_landmarks:
            lh = res.pose_landmarks.landmark[23]
            rh = res.pose_landmarks.landmark[24]
            hx = (lh.x + rh.x) / 2
            hy = (lh.y + rh.y) / 2
            ls = res.pose_landmarks.landmark[11]
            rs = res.pose_landmarks.landmark[12]
            sd = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) or 1
            for lm in res.pose_landmarks.landmark:
                feats.extend([(lm.x - hx) / sd, (lm.y - hy) / sd])
        else:
            feats.extend([0.0] * 66)
            tracking_status = "Missing Pose"

        # Hands (84)
        if not res.right_hand_landmarks and not res.left_hand_landmarks:
            tracking_status = "Missing Hands"

        for hand in [res.right_hand_landmarks, res.left_hand_landmarks]:
            if hand:
                wr = hand.landmark[0]
                md = max(np.sqrt((lm.x - wr.x) ** 2 + (lm.y - wr.y) ** 2)
                         for lm in hand.landmark) or 1
                for lm in hand.landmark:
                    feats.extend([(lm.x - wr.x) / md, (lm.y - wr.y) / md])
            else:
                feats.extend([0.0] * 42)

        seq.append(feats)
        if len(seq) > SEQ_LEN:
            seq.pop(0)

        # Predict on every 5th frame (only when detection is active)
        if detection_active and custom_model is not None and fc % 5 == 0:
            inp = np.array([feats])  # shape (1, 150)
            pred = custom_model.predict(inp, verbose=0)[0]
            conf = float(np.max(pred))
            word = custom_labels[int(np.argmax(pred))]

            with frame_lock:
                if conf > 0.35:
                    current_word = word
                else:
                    current_word = f"? {word} ({int(conf*100)}%)"
        elif not detection_active:
            with frame_lock:
                current_word = ""

        # --- UI Overlay ---
        with frame_lock:
            word_display = current_word

        # Draw status info with better messaging
        if tracking_status == "Missing Hands":
            status_color = (0, 100, 255)  # Orange
            status_msg = "Move back - show your HANDS!"
        elif tracking_status == "Missing Pose":
            status_color = (0, 100, 255)
            status_msg = "Move back - show shoulders & hands!"
        else:
            status_color = (0, 255, 255)  # Cyan
            status_msg = "Ready - perform a sign!"

        # Bright background for status
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, status_msg, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if word_display:
            # Semi-transparent background for text at the TOP
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            color = (0, 255, 100) if not word_display.startswith("?") else (0, 200, 255)
            cv2.putText(frame, word_display.upper(), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        # Status indicator (green dot = running)
        cv2.circle(frame, (w - 20, 20), 10, (0, 255, 0), -1)
        cv2.putText(frame, "LIVE AI", (w - 85, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buf.tobytes()

    holistic.close()
    cap.release()


# Start background thread immediately when module loads
_cam_thread = threading.Thread(target=camera_loop, daemon=True)
_cam_thread.start()
print("[INFO] Camera thread started")

# =========================
# Flask Routes
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


@app.route("/get_gesture")
def get_gesture():
    with frame_lock:
        return jsonify({"word": current_word})


@app.route("/video_feed")
def video_feed():
    def stream():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)  # ~30 FPS
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
            audio = rec.listen(source, timeout=5, phrase_time_limit=5)
        text = rec.recognize_google(audio)
        word = text.lower().strip()

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT gesture_file FROM gestures WHERE word=?", (word,))
            row = cur.fetchone()
            conn.close()
        except Exception:
            row = None

        if row:
            return jsonify({"text": text, "gesture": f"/static/gestures/{row[0]}"})

        gestures = [f"/static/gestures/{c}.gif" for c in word if c.isalpha()]
        if gestures:
            return jsonify({"text": text, "gestures": gestures})
        return jsonify({"text": text, "gesture": "/static/gestures/not_found.gif"})

    except sr.WaitTimeoutError:
        return jsonify({"text": "Timeout - please try again", "gesture": "/static/gestures/not_found.gif"})
    except sr.UnknownValueError:
        return jsonify({"text": "Could not understand audio", "gesture": "/static/gestures/not_found.gif"})
    except Exception as e:
        return jsonify({"text": f"Error: {str(e)}", "gesture": "/static/gestures/not_found.gif"})


if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)