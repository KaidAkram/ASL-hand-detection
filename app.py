import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from PIL import Image
import Levenshtein

st.title("Real-Time Gesture Recognition")
st.write("This app captures hand gestures to predict words.")


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
words_list = ["he", "ho", "lo", "oh", "eh", "hoe", "hel", "ole", "leo", "hole", "heel", "hell", "hello"]


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


labels_dict = {0: 'H', 1: 'E', 2: 'L', 3: 'O', 4: 'end'}
cooldown_duration = 1.5  
word = []
last_predicted_character = None
last_added_time = 0
  


def speak_word(word):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    for voice in engine.getProperty('voices'):
        if "English" in voice.name:
            engine.setProperty('voice', voice.id)
            break
    engine.say(word)
    engine.runAndWait()

def speak_word_async(word):
    threading.Thread(target=speak_word, args=(word,)).start()


def closest_word_jw(word, words_list):
    closest_match = None
    highest_similarity = 0
    for w in words_list:
        similarity = Levenshtein.jaro_winkler(word, w)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_match = w
    return closest_match


if "running" not in st.session_state:
    st.session_state.running = False


frame_placeholder = st.empty()  
recognized_word_placeholder = st.empty()  
history_placeholder = st.empty()  


if st.button("Start Gesture Recognition" ,key="start"):
    st.session_state.running = True


if st.button("Stop Gesture Recognition" ,key="stop"):
    st.session_state.running = False


cap = cv2.VideoCapture(0)  


corrected_word_history = []
if st.session_state.running:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame.")
            break
        frame = cv2.flip(frame, 1)

       
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

            
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
            )

            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                
            data_aux = np.asarray(data_aux)
            if data_aux.shape[0] == 42:
                prediction_probs = model.predict_proba([data_aux])
                predicted_index = np.argmax(prediction_probs, axis=1)[0]
                probability = prediction_probs[0][predicted_index]

                predicted_character = None
                if probability >= 0.9:
                    predicted_character = labels_dict[predicted_index]
                    current_time = time.time()

                    
                    if predicted_character == 'end':
                        if word:
                            recognized_word = ''.join(word)
                            corrected_word = closest_word_jw(recognized_word.lower(), words_list)
                            corrected_word_history.append(corrected_word)  
                            speak_word_async(corrected_word)

                            
                            recognized_word_placeholder.markdown(
                                f"<h2 style='text-align: center; color: white;'>{corrected_word}</h2>", 
                                unsafe_allow_html=True
                            )

                        word = []  
                        last_predicted_character = None  
                    else:
                        
                        if predicted_character == last_predicted_character:
                            if current_time - last_added_time > cooldown_duration:
                                word.append(predicted_character)
                                last_added_time = current_time
                        else:
                            word.append(predicted_character)
                            last_predicted_character = predicted_character
                            last_added_time = current_time

                
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        
        
        frame_placeholder.image(frame, channels="BGR")
        history_placeholder.write(", ".join(corrected_word_history))
else:
    cap.release()
    st.write("Gesture Recognition Stoped")


