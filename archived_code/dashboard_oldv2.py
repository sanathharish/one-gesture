# ui/dashboard.py
import sys
import os
import cv2
import time
import streamlit as st
from collections import deque

# --------------------------------------------------------------------
# PATH FIX (ensures core/ imports work correctly)
# --------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Gesture Engine Modules
from core.detection import HandDetector
from core.features import FeatureExtractor, Smoother, TemporalFingerStabilizer
from core.gestures import GestureEngine


# --------------------------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="One-Gesture Dashboard",
    layout="centered",
    page_icon="🖐️",
)

# Title
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:-10px;">🖐️ One-Gesture</h1>
    <p style="text-align:center; font-size:18px; opacity:0.75;">
        Real-time gesture recognition with stability filters
    </p>
    """,
    unsafe_allow_html=True
)

# Debug toggle
show_debug = st.toggle("🔧 Show Debug Info", value=False)

# Video placeholder (centered)
video_box = st.container()
video_placeholder = video_box.empty()

# Debug section container
if show_debug:
    st.markdown("---")
    st.subheader("📊 Debug Information")
    col1, col2 = st.columns(2)

    static_placeholder = col1.empty()
    dynamic_placeholder = col1.empty()
    finger_count_placeholder = col2.empty()
    fingers_up_placeholder = col2.empty()
    fps_placeholder = st.empty()

    st.subheader("📝 Recent Actions")
    actions_placeholder = st.empty()
else:
    static_placeholder = dynamic_placeholder = finger_count_placeholder = None
    fingers_up_placeholder = fps_placeholder = None
    actions_placeholder = None

# History of detected actions
recent_actions = deque(maxlen=10)


# --------------------------------------------------------------------
# ENGINE INITIALIZATION
# --------------------------------------------------------------------
detector = HandDetector()
extractor = FeatureExtractor()
gesture_engine = GestureEngine(buffer_len=20)

smoother = Smoother(window_size=7)
finger_stabilizer = TemporalFingerStabilizer(history_size=7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Webcam not available. Check privacy settings.")
    st.stop()

prev_time = 0


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    # Detect hand
    results = detector.detect(frame)
    frame = detector.draw(frame, results)

    static_gesture = "None"
    dynamic_gesture = "None"
    finger_count = 0
    finger_names = []

    # ------------------------------------------------------------
    # PROCESS HAND
    # ------------------------------------------------------------
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Relative landmark smoothing
        raw_lms = extractor.landmark_to_relative(hand.landmark)
        rel_lms = smoother.smooth_points(raw_lms)

        # Finger stability
        raw_fingers = extractor.finger_states(hand.landmark)
        finger_states = finger_stabilizer.update(raw_fingers)

        # Finger count + names
        finger_count, finger_names = extractor.finger_count_and_names(finger_states)

        # Track centroid for dynamic gestures
        cx = sum([p[0] for p in rel_lms]) / len(rel_lms)
        cy = sum([p[1] for p in rel_lms]) / len(rel_lms)
        gesture_engine.update_history((cx, cy))

        # Static + dynamic gestures
        sg = gesture_engine.detect_static(rel_lms, finger_states)
        static_gesture = gesture_engine.filter_static(sg)
        dynamic_gesture = gesture_engine.detect_dynamic()

        # Log actions
        if static_gesture not in ["None", None]:
            recent_actions.append(static_gesture)
        if dynamic_gesture not in ["None", None]:
            recent_actions.append(dynamic_gesture)

        # Bounding box
        h, w, _ = frame.shape
        x1, y1, x2, y2 = extractor.hand_bounding_box(hand.landmark)
        cv2.rectangle(
            frame,
            (int(x1 * w), int(y1 * h)),
            (int(x2 * w), int(y2 * h)),
            (0, 255, 0),
            2
        )

        # Text overlay
        cv2.putText(frame, static_gesture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 230, 0), 2)
        if dynamic_gesture:
            cv2.putText(frame, dynamic_gesture, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 200), 2)

    # ------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Render to Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, use_container_width=True)

    # ------------------------------------------------------------
    # UPDATE DEBUG UI (ONLY IF ENABLED)
    # ------------------------------------------------------------
    if show_debug:
        static_placeholder.write(f"**Static Gesture:** {static_gesture}")
        dynamic_placeholder.write(f"**Dynamic Gesture:** {dynamic_gesture}")
        finger_count_placeholder.write(f"**Finger Count:** {finger_count}")
        fingers_up_placeholder.write(f"**Fingers Up:** {', '.join(finger_names) if finger_names else 'None'}")
        fps_placeholder.write(f"**FPS:** {int(fps)}")
        actions_placeholder.write("\n".join(f"- {a}" for a in list(recent_actions)))
