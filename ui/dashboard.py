# ui/dashboard.py
import sys
import os
import cv2
import time
import streamlit as st
from collections import deque
import numpy as np

# Add project root to sys.path so core modules are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import One-Gesture engine modules
from core.detection import HandDetector
from core.features import FeatureExtractor
from core.gestures import GestureEngine


# ---------------- SMOOTHER (NEW) ---------------- #

class Smoother:
    """Smooths list of (x, y) landmark points using moving average."""
    def __init__(self, size=5):
        self.size = size
        self.buffer = deque(maxlen=size)

    def smooth(self, pts):
        self.buffer.append(pts)
        avg = np.mean(self.buffer, axis=0)
        return [(float(a), float(b)) for a, b in avg]


# ---------------- UI SETUP ---------------- #

st.set_page_config(
    page_title="One-Gesture Dashboard",
    layout="wide",
    page_icon="🖐️",
)

st.title("🖐️ One-Gesture Dashboard")
st.write("Real-time hand tracking, gesture detection, and system action preview.")

# Smaller video feed → reduce width to look cleaner
col_video, col_info = st.columns([2, 1])

video_placeholder = col_video.empty()

col_info.subheader("Detected Info")
gesture_placeholder = col_info.empty()
dynamic_placeholder = col_info.empty()
fps_placeholder = col_info.empty()

col_info.subheader("Recent Actions")
actions_placeholder = col_info.empty()

recent_actions = deque(maxlen=10)


# ---------------- ENGINE SETUP ---------------- #

detector = HandDetector()
extractor = FeatureExtractor()
gesture_engine = GestureEngine(buffer_len=10)
smoother = Smoother(size=5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Cannot open camera. Check privacy, firewall, or if another app is using it.")
    st.stop()

prev_time = 0


# ---------------- MAIN LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)
    frame = detector.draw(frame, results)

    static_gesture = "None"
    dynamic_gesture = "None"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Extract raw landmarks → smooth them
        raw_lms = extractor.landmark_to_relative(hand.landmark)
        rel_lms = smoother.smooth(raw_lms)

        finger_states = extractor.finger_states(hand.landmark)
        bbox = extractor.hand_bounding_box(hand.landmark)

        # Centroid → smoothed also
        cx = sum([p[0] for p in rel_lms]) / len(rel_lms)
        cy = sum([p[1] for p in rel_lms]) / len(rel_lms)
        gesture_engine.update_history((cx, cy))

        # Gesture detection (smoothed)
        static_gesture = gesture_engine.detect_static(rel_lms, finger_states)
        dynamic_gesture = gesture_engine.detect_dynamic()

        # Log gestures
        if static_gesture and static_gesture != "None":
            recent_actions.append(static_gesture)
        if dynamic_gesture and dynamic_gesture != "None":
            recent_actions.append(dynamic_gesture)

        # Draw bounding box
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)

        # Overlay gesture labels
        cv2.putText(frame, static_gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 200, 0), 2)
        if dynamic_gesture:
            cv2.putText(frame, dynamic_gesture, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 0, 200), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    # Show frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, use_column_width=True)

    # Update sidebar info
    gesture_placeholder.write(f"**Static Gesture:** {static_gesture}")
    dynamic_placeholder.write(f"**Dynamic Gesture:** {dynamic_gesture}")
    fps_placeholder.write(f"**FPS:** {int(fps)}")

    actions_placeholder.write("\n".join(f"- {a}" for a in list(recent_actions)))
