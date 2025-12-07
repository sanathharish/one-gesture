# ui/dashboard.py
import sys
import os

# 1️⃣ Add project root to sys.path so core/ and utils/ can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2️⃣ Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from core.detection import HandDetector
from core.features import FeatureExtractor
from core.gestures import GestureEngine

# Initialize gesture engine
gesture_engine = GestureEngine(buffer_len=5)

# HandProcessor class for Streamlit-WebRTC
class HandProcessor:
    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()

    def recv(self, frame):
        # Convert video frame to OpenCV BGR
        img = frame.to_ndarray(format="bgr24")
        results = self.detector.detect(img)
        img = self.detector.draw(img, results)

        # Process first hand landmarks if detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            rel_lms = self.extractor.landmark_to_relative(hand_landmarks.landmark)
            finger_states = self.extractor.finger_states(hand_landmarks.landmark)
            bbox = self.extractor.hand_bounding_box(hand_landmarks.landmark)

            # Optional: print landmark info for debugging
            print("Rel:", [tuple(round(x,2) for x in lm) for lm in rel_lms[:5]])
            print("Fingers:", finger_states, "BBox:", [round(b,2) for b in bbox])

        return img

# Streamlit UI
st.title("One-Gesture Dashboard")
st.text("Webcam feed with live hand tracking and gesture detection.")

webrtc_streamer(
    key="one-gesture",
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True},
    async_processing=True,
)

st.text("Press 'q' in the webcam window or stop Streamlit to exit.")
