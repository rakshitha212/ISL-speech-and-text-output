# Indian Sign Language (ISL) Recognition & Synthesis System

Real-time communication bridge for the deaf and hard of hearing, featuring two-way translation between Indian Sign Language and spoken/text output.

---

## Features

### 1. Gesture to Speech & Text (Recognition)
- Real-time webcam capture of ISL gestures using MediaPipe Holistic
- AI4Bharat **INCLUDE Transformer** model for 50-word ISL recognition
- Automatic text-to-speech output when a gesture is locked
- Stability logic with confidence thresholding to prevent flickering predictions
- Hand interpolation for brief occlusions (carries forward last known position)

### 2. Speech to Gesture (Synthesis)
- Voice input via Google Speech Recognition API
- Plays corresponding ISL sign videos from a curated library
- Alphabet finger-spelling fallback for unknown words (A-Z GIFs)

---

## Architecture & Implementation

### Gesture Recognition Pipeline

```
Webcam Frame (30fps)
    │
    ▼
MediaPipe Holistic ─── Pose (25 upper-body landmarks) + Hands (2 × 21 landmarks)
    │                        │
    │                        ▼
    │               Feature Extraction: raw (x,y) × (1920, 1080) = 134 features/frame
    │                        │
    │                        ▼
    │               Sequence Buffer: 60 frames (2 seconds)
    │                        │
    │               ┌────────┴─────────┐
    │               │  Inference every  │
    │               │  5 frames (~6 Hz) │
    │               └────────┬─────────┘
    │                        │
    │                        ▼
    │               INCLUDE Transformer (PyTorch, small variant)
    │               Input: (1, 60, 134) → Output: (1, 50) logits
    │                        │
    │                        ▼
    │               Softmax → Top prediction + confidence
    │                        │
    │                        ▼
    │               Stability logic: 3 consecutive >40% → LOCK word
    │                        │
    ▼                        ▼
OpenCV overlay          Flask JSON API → Frontend
```

### Feature Extraction (134 features per frame)

| Component | Landmarks | Features | Coordinate Scale |
|-----------|-----------|----------|-----------------|
| Upper body pose | 25 (indices 0-24) | 50 | `x × 1920`, `y × 1080` |
| Left hand | 21 | 42 | `x × 1920`, `y × 1080` |
| Right hand | 21 | 42 | `x × 1920`, `y × 1080` |
| **Total** | | **134** | |

This matches the INCLUDE training pipeline where raw MediaPipe coordinates (0-1 range) are scaled to pixel dimensions (1920×1080). No per-frame or per-sequence normalization is applied, consistent with the AI4Bharat/INCLUDE `dataset.py`.

### INCLUDE Transformer Model

- **Source**: [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) — ACM MM 2020
- **Architecture**: Pure PyTorch Transformer (no `transformers` library dependency)
  - Linear embedding: 134 → 256
  - Positional embedding (learned, max 256 positions)
  - 2 BertLayer blocks (self-attention + FFN)
  - Max pooling over sequence dimension
  - Classification head: 256 → 50
- **Weights**: `model/include50_transformer_small.pth` (pre-trained on INCLUDE50 dataset)
- **Labels**: `model/include_labels.json` (50 ISL words → integer indices)

### Inference Optimizations

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Buffer length | 60 frames (2 sec) | INCLUDE model supports up to 256 positions; longer context captures full gestures |
| Inference interval | Every 5 frames (~6 Hz) | Reduces CPU load and prediction noise; 30 Hz inference is unnecessary |
| Hand interpolation | Carry forward up to 10 frames | Matches INCLUDE's NaN interpolation; prevents jarring 0-fills during brief occlusions |
| Confidence threshold | 0.40 | Calibrated for 50-class Transformer (peak observed ~47%) |
| Stability count | 3 consecutive | 3 × 5 frames = 15 frames = 0.5 sec of consistent prediction before locking |
| Cooldown | 2 seconds | Prevents repeated locking of the same word |

### Speech-to-Gesture Pipeline

```
Microphone Audio
    │
    ▼
Google Speech Recognition API → Text
    │
    ▼
SQLite3 lookup (word → gesture file)
    │
    ├── Found → Play MP4/GIF video
    └── Not found → Spell out with A-Z finger-spelling GIFs
```

---

## Supported Words (INCLUDE50 — 50 words)

bank, biglarge, bird, black, boy, brother, car, cellphone, court, cow, death, dog, dry, election, fall, fan, father, girl, good, goodmorning, happy, hat, hello, hot, house, i, it, long, loud, monday, new, paint, pen, priest, quiet, red, shoes, short, smalllittle, storeorshop, summer, teacher, thankyou, time, trainticket, tshirt, white, window, year, youplural

> **Note**: "beautiful" is NOT in the INCLUDE50 vocabulary. Use "happy" instead. The old LSTM model (41 classes) is available as a fallback if the INCLUDE model fails to load.

### ISL Gesture References
- [ISL Dictionary](https://indiansignlanguage.org/) — Images and videos for each sign
- [INCLUDE Dataset (Zenodo)](https://zenodo.org/record/4010759) — Original training videos
- [YouTube ISL Demos](https://www.youtube.com/results?search_query=indian+sign+language+INCLUDE+gesture)

### Camera Guidelines for Best Results
- Stand 1-2 meters from the camera
- Face the camera directly with upper body visible (shoulders to head)
- Ensure good lighting on your face and hands
- Keep both hands visible in the frame at all times
- Perform the full gesture slowly and hold for 1-2 seconds
- Avoid cluttered or backlit backgrounds

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.10, Flask |
| Frontend | HTML5, CSS (Glassmorphism), JavaScript |
| Computer Vision | MediaPipe Holistic, OpenCV |
| Gesture Model | PyTorch (INCLUDE Transformer, 50 classes) |
| Fallback Model | TensorFlow/Keras (LSTM, 41 classes) |
| Speech Engine | Google Speech Recognition API, Web Speech API |
| Database | SQLite3 |

---

## Project Structure

```
ISL-speech-and-text-output/
├── app.py                          # Main Flask application (camera loop, inference, routes)
├── train_model.py                  # LSTM training pipeline (41-class fallback model)
├── process_videos.py               # Video conversion & DB sync utility
├── test_model_load.py              # INCLUDE model integration test script
├── inspect_weights.py              # Model weight inspection utility
├── requirements.txt                # Python dependencies
├── model/
│   ├── include_models.py           # Pure PyTorch Transformer (INCLUDE architecture)
│   ├── include_labels.json         # 50-word label map (word → index)
│   ├── include50_transformer_small.pth  # Pre-trained INCLUDE weights
│   ├── gesture_model_lstm.h5       # Trained LSTM weights (fallback)
│   └── labels.json                 # 41-class LSTM label map
├── dataset/                        # Raw data (CSV landmarks & MOV videos)
├── static/
│   ├── style.css                   # UI styling
│   ├── gestures/                   # ISL sign videos (MP4) and alphabet GIFs
│   └── script.js                   # General frontend logic
├── templates/
│   ├── index.html                  # Home page
│   ├── gesture.html                # Gesture-to-speech page (with supported words panel)
│   ├── speech.html                 # Speech-to-gesture page
│   └── about.html                  # About page
└── database/
    └── isl.db                      # SQLite word → gesture mapping
```

---

## Setup & Installation

### Prerequisites
- **Python 3.10** (tested with 3.10.11)
- **Webcam** (built-in or USB)
- **Microphone** (for speech-to-gesture feature)
- **Windows** (tested on Windows; may work on macOS/Linux with adjustments)

### 1. Clone the Repository
```bash
git clone https://github.com/rakshitha212/ISL-speech-and-text-output.git
cd ISL-speech-and-text-output
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment
```bash
# Windows PowerShell
.\.venv\Scripts\activate

# Windows CMD
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `Flask` — Web server
- `opencv-python` — Camera capture & frame processing
- `mediapipe` — Pose and hand landmark extraction
- `torch` — PyTorch for INCLUDE Transformer inference
- `tensorflow` — TensorFlow/Keras for LSTM fallback model
- `numpy` — Numerical operations
- `SpeechRecognition` — Voice input
- `PyAudio` — Microphone access

### 5. Verify Model Files
Ensure these files exist in the `model/` directory:
- `include50_transformer_small.pth` — INCLUDE Transformer weights
- `include_labels.json` — 50-word label map
- `include_models.py` — Transformer architecture definition

If the INCLUDE model files are missing, the app will fall back to the LSTM model (41 classes).

### 6. Run the Application
```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 7. Using the Gesture Recognition
1. Navigate to the **Gesture to Speech** page
2. Click **Enable Camera**
3. Wait for the buffer to fill (shows "Buffering... 60/60")
4. Perform an ISL gesture from the supported words list
5. The system will show "Analyzing... word (XX%)" as confidence builds
6. When confidence is consistently >40%, the word is **LOCKED** and spoken aloud
7. After a 2-second cooldown, the system resets for the next gesture

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/gesture` | GET | Gesture-to-speech page |
| `/speech` | GET | Speech-to-gesture page |
| `/about` | GET | About page |
| `/video_feed` | GET | MJPEG video stream from webcam |
| `/get_gesture` | GET | Current prediction `{word, status}` |
| `/current_prediction` | GET | Locked word `{prediction}` |
| `/toggle_detection` | POST | Start/stop camera `{active: bool}` |
| `/detection_status` | GET | Camera active status `{active}` |
| `/supported_words` | GET | List of 50 INCLUDE words `{words, count}` |
| `/speech_to_sign` | POST | Voice → ISL gesture `{text, gesture/gestures}` |

---

## Debugging

When the app is running, watch the console for `[DEBUG]` lines:
```
[DEBUG] pred=happy conf=0.623 | L=Y R=Y | top3: happy=0.62, good=0.12, hello=0.05
```

- **pred** — Top predicted word
- **conf** — Confidence (softmax probability)
- **L/R** — Left/Right hand detection status (Y=detected, N=missing)
- **top3** — Top 3 predictions with probabilities

If confidence stays low (<20%), check:
1. Are both hands visible? (`L=Y R=Y`)
2. Is the gesture from the supported 50 words?
3. Is lighting adequate?
4. Is the upper body (shoulders to head) visible?

---

## Datasets

| Dataset | Use | Link |
|---------|-----|------|
| INCLUDE50 | Gesture recognition (Transformer, 50 words) | [Zenodo](https://zenodo.org/record/4010759) |
| ISL Words with Landmarks | Gesture recognition (LSTM, 41 words) | [Kaggle](https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks) |

---

## License

 2025 Indian Sign Language Detection Project.
