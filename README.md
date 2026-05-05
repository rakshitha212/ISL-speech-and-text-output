# ISL-speech-and-text-output
📌 Indian Sign Language (ISL) Real-Time System
🚀 Project Overview

This project is a real-time Indian Sign Language (ISL) translation system that provides:

✋ Gesture to Word – Converts hand gestures into text
🎤 Speech to Text with Sign Language Output – Converts speech into text and displays corresponding ISL gestures

The system is designed for high accuracy, fast response, and real-time interaction.

🎯 Core Functionalities
✋ 1. Gesture to Word (Real-Time Recognition)
🔍 Description

This module captures hand gestures using a webcam and converts them into meaningful words using a trained machine learning model.

🔁 Workflow
Webcam → Hand Detection → Landmark Extraction → ML Model → Word Output
⚙️ How It Works
Uses MediaPipe to detect 21 hand landmarks
Extracts key points from hand gestures
Feeds data into a trained CNN model
Predicts gesture and maps it to a word
✅ Features
Real-time gesture detection
High accuracy prediction
Supports ISL alphabets & predefined words
Works with live webcam feed
🎤 2. Speech to Text with Sign Language Output
🔍 Description

This module converts spoken language into text and then displays the corresponding ISL gesture visually.

🔁 Workflow
Speech Input → Speech Recognition → Text Output → Gesture Mapping → Sign Display
⚙️ How It Works
Captures voice using microphone
Converts speech → text using SpeechRecognition API
Matches recognized text with gesture database
Displays ISL gesture as image/GIF/video
✅ Features
Fast speech-to-text conversion
Real-time gesture visualization
Supports word-based ISL mapping
Easy and interactive UI
🏗️ Tech Stack
Component	Technology
Backend	Python (Flask)
Frontend	HTML, CSS, JavaScript
Gesture Model	MediaPipe Gesture Recognizer
Speech Engine	Google Speech Recognition API
Database	SQLite
📂 Project Structure
ISL_Project/
│
├── app.py
│
├── model/
│   └── gesture_model.h5
│
├── static/
│   ├── css/
│   ├── js/
│   ├── gestures/        # ISL gesture images/GIFs
│
├── templates/
│   ├── index.html
│   ├── gesture.html
│   ├── speech.html
│
├── database/
│   └── isl.db
│
├── utils/
│   ├── gesture_module.py
│   ├── speech_module.py
│
├── requirements.txt
└── README.md
## ⚙️ Setup and Execution

Follow these steps to get the project running locally:

### 1️⃣ Prepare the Environment
Ensure you have Python installed. Then, create and activate a virtual environment:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```

### 2️⃣ Install Dependencies
Install all required libraries using the requirements file:
```powershell
pip install -r requirements.txt
```

### 3️⃣ Initialize the Database
Run the initialization script to set up the SQLite database and sample gesture mappings:
```powershell
python init_db.py
```

### 4️⃣ Run the Application
Launch the Flask development server:
```powershell
python app.py
```

### 5️⃣ Access the System
Open your web browser and navigate to:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

🧠 Model Details

### ✋ Gesture Recognition
The system uses the **MediaPipe Gesture Recognizer** task. This model provides real-time classification of hand gestures into distinct categories.
- **Base Model**: MediaPipe Gesture Recognizer (float16)
- **Input**: Live webcam feed / RGB image
- **Output**: Gesture labels (Hello, Thanks, Yes, No, I Love You)

### 🎤 Speech to Sign
The system utilizes the **Google Speech Recognition** engine to process voice input.
- **Processing**: Speech is converted to text, which is then mapped to its corresponding ISL gesture in the database.
- **Output**: Visual sign language representation (GIF/Image).

⚡ Real-Time Performance
Optimized frame processing
Lightweight ML model
Fast API responses
Async frontend communication
🗄️ Database (SQLite)
Gesture Mapping Table
id	word	gesture_file
1	Hello	hello.gif
2	Thanks	thanks.gif
🧪 Requirements
Flask
opencv-python
mediapipe
tensorflow
numpy
sqlite3
SpeechRecognition
pyaudio
