# ASL-hand-detection
A real-time hand gesture recognition app that translates ASL gestures into spoken words


## Overview
This project is a **real-time hand gesture recognition app** that translates hand gestures into words. Using computer vision and machine learning techniques, the app recognizes American Sign Language (ASL) gestures for the letters **H, E, L, O**. It then assembles these gestures into words, performs spell correction, and provides audio feedback using text-to-speech.

This project has been a deep dive into the world of AI and computer vision, leveraging technologies like **OpenCV**, **MediaPipe**, **Random Forest Classification**, **Levenshtein distance** for spelling correction, and **pyttsx3** for text-to-speech functionality. The user interface is built with **Streamlit**, making it user-friendly and interactive.

## Key Features
- **Real-Time Gesture Capture**: Uses webcam input to capture and process hand gestures.
- **Character Prediction & Word Assembly**: Classifies ASL gestures (H, E, L, O) using a **Random Forest** classifier.
- **Spelling Correction**: Applies **Levenshtein distance** to correct any mispredictions and ensure accuracy.
- **Audio Feedback**: Uses **pyttsx3** for text-to-speech, pronouncing the predicted word.
- **User-Friendly Interface**: Built with **Streamlit** to provide an intuitive and easy-to-use experience.

## Requirements
Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- MediaPipe
- Streamlit
- scikit-learn
- pyttsx3
- Levenshtein

You can install the required packages using `pip`:

```bash
pip install opencv-python mediapipe streamlit scikit-learn pyttsx3 python-Levenshtein
```

## How to Run
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/ASL-hand-detection.git
````
Navigate to the project directory:
```bash
cd ASL-hand-detection.git
````

Run the Streamlit app:
```bash
streamlit run app.py
````
The app will open in your browser. You can start making hand gestures in front of the webcam. The app will recognize the ASL gestures for H, E, L, O and display the translated word on the screen, along with audio feedback.

## Technologies Used
  - OpenCV: For real-time video capture and processing.
  - MediaPipe: For hand tracking and recognition.
  - Streamlit: For building a simple and interactive user interface.
  - Random Forest Classifier: For gesture classification.
  - Levenshtein: For spell correction based on string similarity.
  - pyttsx3: For converting text to speech.





