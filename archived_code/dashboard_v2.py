# ui/dashboard.py
import sys
import os
import cv2
import streamlit as st
from core.detection import HandDetector
from core.features import FeatureExtractor
from core.gestures import GestureEngine

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize gesture engine
gesture_engine = GestureEngine(buffer_len=5)

# Initialize Hand Detector and Feature Extractor
detector = HandDetector()
extractor = FeatureExtractor()

st.title("One-Gesture Dashboard")
st.text("Webcam feed with live hand tracking and gesture detection.")

# Video display placeholder
video_placeholder = st.image([])

# OpenCV capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # force DirectShow backend on Windows

if not cap.isOpened():
    st.error("Cannot open camera")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        # Flip frame horizontally for natural selfie view
        frame = cv2.flip(frame, 1)

        # Detect hands
        results = detector.detect(frame)
        frame = detector.draw(frame, results)

        # Process first hand if available
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            rel_lms = extractor.landmark_to_relative(hand_landmarks.landmark)
            finger_states = extractor.finger_states(hand_landmarks.landmark)
            bbox = extractor.hand_bounding_box(hand_landmarks.landmark)

            # Debug prints
            print("Rel:", [tuple(round(x,2) for x in lm) for lm in rel_lms[:5]])
            print("Fingers:", finger_states, "BBox:", [round(b,2) for b in bbox])

        # Convert BGR → RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb)
