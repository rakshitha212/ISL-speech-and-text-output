document.addEventListener("DOMContentLoaded", function () {
    // Mobile menu toggle
    const hamburger = document.querySelector(".hamburger");
    const navLinksContainer = document.querySelector(".nav-links");
  
    hamburger.addEventListener("click", function () {
      navLinksContainer.classList.toggle("active");
    });
  
    // Feature cards animation on hover
    const featureCards = document.querySelectorAll(".feature-card");
  
    featureCards.forEach((card) => {
      card.addEventListener("mouseover", function () {
        this.style.backgroundColor = "#f8f9fa";
      });
  
      card.addEventListener("mouseout", function () {
        this.style.backgroundColor = "#ecf0f1";
      });
    });
  
    // Handle conversion page functionality
    if (document.getElementById("gesture-to-text-page")) {
      const enableCameraBtn = document.getElementById("enable-camera-btn");
      const cameraContainer = document.getElementById("camera-container");
      const placeholderContainer = document.getElementById("placeholder-container");
      const videoFeed = document.getElementById("video-feed");
      const textResult = document.getElementById("text-result");
      
      let isCameraEnabled = false;
      let predictionInterval;

      enableCameraBtn.addEventListener("click", function() {
        if (!isCameraEnabled) {
          // Enable Camera
          console.log("Attempting to connect to camera stream at http://127.0.0.1:5000/video_feed...");
          videoFeed.src = "http://127.0.0.1:5000/video_feed";
          
          videoFeed.onerror = function() {
            console.error("FAILED to load video stream. This is likely a CORS or security block.");
            alert("Error: Unable to connect to the camera stream. Please check your Python terminal for errors and ensure the server is running on port 5000.");
          };

          videoFeed.onload = function() {
            console.log("SUCCESS: Video stream connected.");
          };

          cameraContainer.style.display = "block";
          placeholderContainer.style.display = "none";
          enableCameraBtn.textContent = "Disable Camera";
          isCameraEnabled = true;

          // Start polling for predictions
          predictionInterval = setInterval(async () => {
            try {
              const response = await fetch("http://127.0.0.1:5000/current_prediction");
              if (!response.ok) throw new Error("Server returned " + response.status);
              
              const data = await response.json();
              if (data.prediction) {
                textResult.textContent = `Detected: ${data.prediction}`;
              }
            } catch (error) {
              console.error("Prediction fetch error:", error);
            }
          }, 1000);
        } else {
          // Disable Camera
          console.log("Shutting down camera stream...");
          videoFeed.onerror = null; // Prevent error popup when clearing source
          videoFeed.src = "";
          cameraContainer.style.display = "none";
          placeholderContainer.style.display = "block";
          enableCameraBtn.textContent = "Enable Camera";
          isCameraEnabled = false;
          clearInterval(predictionInterval);
          textResult.textContent = "Your detected text will appear here...";
        }
      });
    }
  
    if (document.getElementById("gesture-to-speech-page")) {
      const startDetectionBtn = document.getElementById("start-detection-btn");
      const cameraContainerSpeech = document.getElementById("camera-container-speech");
      const placeholderContainerSpeech = document.getElementById("placeholder-container-speech");
      const videoFeedSpeech = document.getElementById("video-feed-speech");
      const speechResult = document.getElementById("speech-result");
      const playAudioBtn = document.getElementById("play-audio-btn");
      
      let isDetectionEnabled = false;
      let speechInterval;
      let lastSpokenText = "";

      const speakText = (text) => {
        if (!text || text === "None") return;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9; // Slightly slower for clarity
        window.speechSynthesis.speak(utterance);
      };

      startDetectionBtn.addEventListener("click", function() {
        if (!isDetectionEnabled) {
          // Enable Detection
          console.log("Starting Gesture to Speech detection...");
          videoFeedSpeech.src = "http://127.0.0.1:5000/video_feed";
          
          videoFeedSpeech.onerror = function() {
            alert("Error: Unable to connect to camera stream on port 5000.");
            videoFeedSpeech.onerror = null;
          };

          cameraContainerSpeech.style.display = "block";
          placeholderContainerSpeech.style.display = "none";
          startDetectionBtn.textContent = "Stop Detection";
          isDetectionEnabled = true;

          // Start polling for predictions and speak them
          speechInterval = setInterval(async () => {
            try {
              const response = await fetch("http://127.0.0.1:5000/current_prediction");
              const data = await response.json();
              if (data.prediction && data.prediction !== lastSpokenText) {
                speechResult.textContent = `Detected: ${data.prediction}`;
                speakText(data.prediction);
                lastSpokenText = data.prediction;
              } else if (!data.prediction) {
                speechResult.textContent = "Detected: None";
              }
            } catch (error) {
              console.error("Speech detection error:", error);
            }
          }, 1500); // Slightly longer interval for speech stability
        } else {
          // Disable Detection
          videoFeedSpeech.onerror = null;
          videoFeedSpeech.src = "";
          cameraContainerSpeech.style.display = "none";
          placeholderContainerSpeech.style.display = "block";
          startDetectionBtn.textContent = "Start Detection";
          isDetectionEnabled = false;
          clearInterval(speechInterval);
          speechResult.textContent = "Detected: None";
          lastSpokenText = "";
        }
      });

      playAudioBtn.addEventListener("click", function() {
        if (lastSpokenText) {
          speakText(lastSpokenText);
        }
      });
    }
  
    if (document.getElementById("speech-to-gesture-page")) {
      const micBtn = document.getElementById("mic-btn");
      const micStatus = document.getElementById("mic-status");
      const recognizedTextLabel = document.getElementById("recognized-text");
      const gestureImgDisplay = document.getElementById("gesture-img-display");
      const gesturePlaceholder = document.getElementById("gesture-placeholder");

      if (!micBtn) {
        // New speech.html uses inline startSpeech() - nothing to bind here
        return;
      }
      
      let recognition;
      let isListening = false;
      let sequenceTimeout;

      // Initialize Speech Recognition
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = function() {
          micStatus.textContent = "Listening... Speak clearly into your microphone.";
          micBtn.textContent = "Stop Listening";
          micBtn.style.backgroundColor = "var(--secondary)";
          isListening = true;
        };

        recognition.onresult = function(event) {
          const transcript = event.results[0][0].transcript.toUpperCase();
          recognizedTextLabel.textContent = `Recognized: "${transcript}"`;
          playGestureSequence(transcript);
        };

        recognition.onerror = function(event) {
          console.error("Speech recognition error:", event.error);
          micStatus.textContent = "Error: " + event.error;
          stopListening();
        };

        recognition.onend = function() {
          stopListening();
        };
      } else {
        micStatus.textContent = "Your browser does not support Speech Recognition. Try Chrome or Edge.";
        micBtn.disabled = true;
      }

      function stopListening() {
        if (recognition) recognition.stop();
        micBtn.textContent = "Enable Microphone";
        micBtn.style.backgroundColor = "var(--accent)";
        isListening = false;
        if (micStatus.textContent === "Listening...") {
          micStatus.textContent = "Click the button to start again.";
        }
      }

      async function playGestureSequence(text) {
        clearTimeout(sequenceTimeout);
        const chars = text.replace(/[^A-Z]/g, '').split('');
        if (chars.length === 0) return;

        gesturePlaceholder.style.display = "none";
        gestureImgDisplay.style.display = "block";

        for (let i = 0; i < chars.length; i++) {
          const char = chars[i].toLowerCase();
          gestureImgDisplay.src = `speech to sign/${char}.jpg`;
          recognizedTextLabel.innerHTML = `Signing: <strong>${text}</strong> (Letter: ${chars[i]})`;
          await new Promise(resolve => {
            sequenceTimeout = setTimeout(resolve, 1000);
          });
        }

        gestureImgDisplay.style.display = "none";
        gesturePlaceholder.style.display = "block";
        recognizedTextLabel.textContent = `Recognized: "${text}" (Done)`;
        micStatus.textContent = "Sequence complete. Click button to speak again.";
      }

      micBtn.addEventListener("click", function() {
        if (!isListening) {
          recognition.start();
        } else {
          stopListening();
        }
      });
    }
  
    // Close mobile menu when clicking anywhere on the page
    document.addEventListener("click", function(event) {
      if (
        !event.target.closest(".hamburger") &&
        !event.target.closest(".nav-links") &&
        navLinksContainer.classList.contains("active")
      ) {
        navLinksContainer.classList.remove("active");
      }
    });
  });