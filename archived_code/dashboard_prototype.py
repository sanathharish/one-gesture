# ui/dashboard.py
import sys
import os
import cv2
import time
import streamlit as st
import numpy as np
from collections import deque

# ----------------------------------------------------------------------------
# FIX IMPORT PATH
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Core imports
from core.detection import HandDetector
from core.features import FeatureExtractor, Smoother, TemporalFingerStabilizer
from core.gestures import GestureEngine


# ----------------------------------------------------------------------------
# STREAMLIT SETUP
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="One-Gesture Dashboard",
    page_icon="🖐️",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>🖐️ One-Gesture Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Real-time Hand Tracking & Gesture Recognition</p>",
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# SAFE DEBUG TOGGLE  (AVOIDS STREAMLIT RERUN CRASHES)
# ----------------------------------------------------------------------------
if "debug" not in st.session_state:
    st.session_state.debug = False

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    debug_toggle = st.toggle("Show Debug Info", value=st.session_state.debug)
    st.session_state.debug = debug_toggle

debug = st.session_state.debug

# ----------------------------------------------------------------------------
# UI SECTIONS (STATIC, OUTSIDE LOOP — VERY IMPORTANT)
# ----------------------------------------------------------------------------
video_placeholder = st.empty()

debug_box = st.container()
debug_box_static = debug_box.empty()
debug_box_dynamic = debug_box.empty()
debug_box_fingers = debug_box.empty()
debug_box_finger_names = debug_box.empty()
debug_box_fps = debug_box.empty()

st.subheader("Recent Actions")
actions_placeholder = st.empty()

recent_actions = deque(maxlen=12)

# ----------------------------------------------------------------------------
# ENGINE SETUP
# ----------------------------------------------------------------------------
detector = HandDetector()
extractor = FeatureExtractor()
gesture_engine = GestureEngine(buffer_len=15)
smoother = Smoother(window_size=5)
finger_stabilizer = TemporalFingerStabilizer(history_size=7)

# ----------------------------------------------------------------------------
# CAMERA INIT (DSHOW FIX FOR WINDOWS)
# ----------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Unable to access webcam. Close other apps using the camera.")
    st.stop()

prev_time = 0

# ----------------------------------------------------------------------------
# MAIN LOOP — NEVER TOUCHES STREAMLIT WIDGETS
# ----------------------------------------------------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)
    frame = detector.draw(frame, results)

    static_gesture = "None"
    dynamic_gesture = "None"
    finger_names = []
    finger_count = 0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # 1. Smooth landmarks
        raw = extractor.landmark_to_relative(hand.landmark)
        rel_lms = smoother.smooth_points(raw)

        # 2. Stabilized finger states
        raw_states = extractor.finger_states(hand.landmark)
        finger_states = finger_stabilizer.update(raw_states)

        # 3. Finger names/count
        finger_count, finger_names = extractor.finger_count_and_names(finger_states)

        # 4. Dynamic movement
        cx = sum(p[0] for p in rel_lms) / len(rel_lms)
        cy = sum(p[1] for p in rel_lms) / len(rel_lms)
        gesture_engine.update_history((cx, cy))

        # 5. Gestures
        sg = gesture_engine.detect_static(rel_lms, finger_states)
        static_gesture = gesture_engine.filter_static(sg)
        dynamic_gesture = gesture_engine.detect_dynamic()

        if static_gesture not in ["None", None]:
            recent_actions.append(static_gesture)
        if dynamic_gesture not in ["None", None]:
            recent_actions.append(dynamic_gesture)

        # 6. Draw bbox
        h, w, _ = frame.shape
        x1, y1, x2, y2 = extractor.hand_bounding_box(hand.landmark)
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0,255,0), 2)

        # Labels on video
        cv2.putText(frame, static_gesture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 230, 0), 3)
        if dynamic_gesture:
            cv2.putText(frame, dynamic_gesture, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 200), 3)

    # ----------------------------------------------------------------------------
    # FPS
    # ----------------------------------------------------------------------------
    curr = time.time()
    fps = 1 / (curr - prev_time) if prev_time else 0
    prev_time = curr

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # ----------------------------------------------------------------------------
    # SHOW VIDEO
    # ----------------------------------------------------------------------------
    video_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    # ----------------------------------------------------------------------------
    # DEBUG INFO (Shown only if debug enabled)
    # ----------------------------------------------------------------------------
    if debug:
        debug_box_static.write(f"**Static Gesture:** {static_gesture}")
        debug_box_dynamic.write(f"**Dynamic Gesture:** {dynamic_gesture}")
        debug_box_fingers.write(f"**Finger Count:** {finger_count}")
        debug_box_finger_names.write(f"**Fingers Up:** {', '.join(finger_names) if finger_names else 'None'}")
        debug_box_fps.write(f"**FPS:** {int(fps)}")
    else:
        debug_box_static.empty()
        debug_box_dynamic.empty()
        debug_box_fingers.empty()
        debug_box_finger_names.empty()
        debug_box_fps.empty()

    # ----------------------------------------------------------------------------
    # RECENT ACTIONS
    # ----------------------------------------------------------------------------
    actions_placeholder.write("\n".join(f"- {a}" for a in list(recent_actions)))
