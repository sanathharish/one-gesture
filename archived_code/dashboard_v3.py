# ui/dashboard.py
import sys
import os
import cv2
import time
import streamlit as st
from collections import deque
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Suppress TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Engine imports
from core.detection import HandDetector
from core.features import FeatureExtractor
from core.gestures import GestureEngine


# ---------------- SMOOTHER ---------------- #

class Smoother:
    """Smooth (x,y) landmark points using moving average."""
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)

    def smooth(self, pts):
        self.buffer.append(pts)
        avg = np.mean(self.buffer, axis=0)
        return [(float(a), float(b)) for a, b in avg]


# ---------------- UI SETUP ---------------- #

st.set_page_config(
    page_title="One-Gesture Dashboard",
    layout="centered",
    page_icon="🖐️",
)

# Center page header
st.markdown(
    "<h1 style='text-align:center;'>🖐️ One-Gesture Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Real-time hand tracking & gesture detection.</p>",
    unsafe_allow_html=True
)

video_placeholder = st.empty()  # Centered & smaller by design

st.subheader("Detected Info")
gesture_placeholder = st.empty()
dynamic_placeholder = st.empty()
fps_placeholder = st.empty()

st.subheader("Recent Actions")
actions_placeholder = st.empty()

recent_actions = deque(maxlen=10)


# ---------------- ENGINE SETUP ---------------- #

detector = HandDetector()
extractor = FeatureExtractor()
gesture_engine = GestureEngine(buffer_len=15)  # smoother tracking
smoother = Smoother(size=5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Cannot open camera. Close Teams/Zoom/OBS or check privacy settings.")
    st.stop()

prev_time = 0


# ---------------- MAIN LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Camera read failed.")
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)
    frame = detector.draw(frame, results)

    static_gesture = "None"
    dynamic_gesture = "None"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Raw → Smoothed landmarks
        raw_lms = extractor.landmark_to_relative(hand.landmark)
        rel_lms = smoother.smooth(raw_lms)

        finger_states = extractor.finger_states(hand.landmark)
        bbox = extractor.hand_bounding_box(hand.landmark)

        # Centroid (smoothed)
        cx = sum([p[0] for p in rel_lms]) / len(rel_lms)
        cy = sum([p[1] for p in rel_lms]) / len(rel_lms)
        gesture_engine.update_history((cx, cy))

        # Dynamic movement fast → suppress static gesture flicker
        if len(gesture_engine.centroid_history) >= 3:
            dx = abs(gesture_engine.centroid_history[-1][0] -
                     gesture_engine.centroid_history[-3][0])
            dy = abs(gesture_engine.centroid_history[-1][1] -
                     gesture_engine.centroid_history[-3][1])
            moving_fast = dx > 0.03 or dy > 0.03
        else:
            moving_fast = False

        # Detect gestures only if stable
        if not moving_fast:
            static_gesture = gesture_engine.detect_static(rel_lms, finger_states)

        dynamic_gesture = gesture_engine.detect_dynamic()

        # Action logs
        if static_gesture not in ["None", None]:
            recent_actions.append(static_gesture)
        if dynamic_gesture not in ["None", None]:
            recent_actions.append(dynamic_gesture)

        # Draw bounding box
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame,
            (int(x1*w), int(y1*h)),
            (int(x2*w), int(y2*h)),
            (0, 255, 0),
            2
        )

        # Overlay gesture labels
        cv2.putText(frame, static_gesture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        if dynamic_gesture:
            cv2.putText(frame, dynamic_gesture, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 200), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Convert frame & display (centered + smaller)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, use_container_width=True)

    # Dashboard side info
    gesture_placeholder.write(f"**Static Gesture:** {static_gesture}")
    dynamic_placeholder.write(f"**Dynamic Gesture:** {dynamic_gesture}")
    fps_placeholder.write(f"**FPS:** {int(fps)}")

    actions_placeholder.write("\n".join(f"- {a}" for a in list(recent_actions)))
