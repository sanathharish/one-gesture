# ui/dashboard.py
import sys
import os
import time
from collections import deque

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ----------------------------------------------------------------------------
# PATH FIX FOR IMPORTS
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from core.detection import HandDetector
from core.features import FeatureExtractor, Smoother, TemporalFingerStabilizer
from core.gestures import GestureEngine


# ----------------------------------------------------------------------------
# PAGE + CYBERPUNK THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="One-Gesture Dashboard",
    page_icon="🖐️",
    layout="centered",
)

st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top, #031019 0%, #02070d 40%, #000 100%) !important;
        color: #eaffff !important;
    }
    .stApp {
        background: radial-gradient(circle at top, #031019 0%, #02070d 40%, #000 100%) !important;
    }
    .main .block-container {
        padding-top: 1rem;
        max-width: 1100px;
    }
    .glass-card {
        background: rgba(0, 30, 50, 0.35);
        backdrop-filter: blur(16px);
        border-radius: 18px;
        border: 1px solid rgba(0, 255, 255, 0.25);
        padding: 0.8rem 1rem;
        box-shadow: 0 0 22px rgba(0,255,255,0.18);
    }
    .og-title {
        text-align: center;
        color: #eaffff;
        font-size: 2.3rem;
        text-shadow: 0 0 16px cyan;
        letter-spacing: 4px;
        margin-bottom: 0;
        text-transform: uppercase;
    }
    .og-subtitle {
        text-align: center;
        color: #b7f8ff;
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    .og-label {
        color: #9fefff;
        font-size: 0.85rem;
    }
    .og-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        text-shadow: 0 0 10px cyan;
    }
    .og-chip {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.1rem;
        border-radius: 12px;
        border: 1px solid rgba(0,255,255,0.25);
        color: #c4faff;
        font-size: 0.78rem;
        background: rgba(0, 180, 255, 0.15);
    }
    .og-actions-log {
        max-height: 150px;
        overflow-y: auto;
        font-family: monospace;
        color: #c5f8ff;
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] {
        background: rgba(5, 15, 30, 0.95);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
        backdrop-filter: blur(18px);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #e0f6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="og-title">ONE-GESTURE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="og-subtitle">Cyberpunk Hand, Finger & Swipe Control (WebRTC Hybrid)</div>',
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# VIDEO PROCESSOR (WEBRTC) — CORE HYBRID ENGINE
# ----------------------------------------------------------------------------
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        # Core engines
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
        self.gesture_engine = GestureEngine(buffer_len=20)
        self.smoother = Smoother(window_size=6)
        self.finger_stabilizer = TemporalFingerStabilizer(history_size=7)

        # For UI / debug
        self.static_gesture = "None"
        self.dynamic_gesture = "None"
        self.finger_names = []
        self.finger_count = 0
        self.fps = 0
        self.recent_actions = deque(maxlen=12)

        self._prev_time = time.time()

    def recv(self, frame):
        # Convert WebRTC VideoFrame -> BGR image
        img = frame.to_ndarray(format="bgr24")

        # Hand detection
        results = self.detector.detect(img)
        img = self.detector.draw(img, results)

        static_gesture = "None"
        dynamic_gesture = "None"
        finger_names = []
        finger_count = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Smooth relative landmarks
            raw_lms = self.extractor.landmark_to_relative(hand.landmark)
            rel_lms = self.smoother.smooth_points(raw_lms)

            # Stabilized finger states
            raw_states = self.extractor.finger_states(hand.landmark)
            finger_states = self.finger_stabilizer.update(raw_states)

            # Finger info
            finger_count, finger_names = self.extractor.finger_count_and_names(
                finger_states
            )

            # Dynamic movement
            cx = sum(p[0] for p in rel_lms) / len(rel_lms)
            cy = sum(p[1] for p in rel_lms) / len(rel_lms)
            self.gesture_engine.update_history((cx, cy))

            # Gestures
            sg = self.gesture_engine.detect_static(rel_lms, finger_states)
            static_gesture = self.gesture_engine.filter_static(sg)
            dynamic_gesture = self.gesture_engine.detect_dynamic()

            # Action log
            if static_gesture not in ["None", None]:
                self.recent_actions.append(static_gesture)
            if dynamic_gesture not in ["None", None]:
                self.recent_actions.append(dynamic_gesture)

            # Draw bounding box
            h, w, _ = img.shape
            x1, y1, x2, y2 = self.extractor.hand_bounding_box(hand.landmark)
            cv2.rectangle(
                img,
                (int(x1 * w), int(y1 * h)),
                (int(x2 * w), int(y2 * h)),
                (0, 255, 255),
                2,
            )

            # Overlay gestures
            cv2.putText(
                img,
                static_gesture,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 200),
                2,
            )
            if dynamic_gesture:
                cv2.putText(
                    img,
                    dynamic_gesture,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

        # FPS
        now = time.time()
        fps = 1 / (now - self._prev_time) if self._prev_time else 0
        self._prev_time = now

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        # Update attributes for UI
        self.static_gesture = static_gesture
        self.dynamic_gesture = dynamic_gesture
        self.finger_count = finger_count
        self.finger_names = finger_names
        self.fps = int(fps)

        # Return back to WebRTC as VideoFrame (BGR24 is fine)
        return img


# ----------------------------------------------------------------------------
# SIDEBAR – SETTINGS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    debug = st.toggle("🔧 Show Debug Panel", value=True)
    st.markdown(
        "WebRTC is used for camera streaming. "
        "Make sure to allow camera permission in the browser.",
    )


# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
webrtc_ctx = webrtc_streamer(
    key="one-gesture-webrtc",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DEBUG / INFO PANEL (reads attributes from processor)
# ----------------------------------------------------------------------------
if debug:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    if webrtc_ctx and webrtc_ctx.video_processor:
        vp: GestureProcessor = webrtc_ctx.video_processor

        with col1:
            st.markdown(
                f"<div class='og-label'>Static Gesture</div>"
                f"<div class='og-value'>{vp.static_gesture}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='og-label'>Dynamic Gesture</div>"
                f"<div class='og-value'>{vp.dynamic_gesture}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"<div class='og-label'>Finger Count</div>"
                f"<div class='og-value'>{vp.finger_count}</div>",
                unsafe_allow_html=True,
            )

            if vp.finger_names:
                chips = "".join(
                    f"<span class='og-chip'>{name}</span>" for name in vp.finger_names
                )
            else:
                chips = "<span class='og-chip'>None</span>"

            st.markdown(
                f"<div class='og-label'>Fingers Up</div>{chips}",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='og-label'>Frame Rate</div>"
            f"<div class='og-value'>{vp.fps} FPS</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Recent actions
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            "<div class='og-label'>Recent Actions</div>",
            unsafe_allow_html=True,
        )
        if vp.recent_actions:
            log_html = (
                "<div class='og-actions-log'>"
                + "<br>".join(f"• {a}" for a in list(vp.recent_actions))
                + "</div>"
            )
        else:
            log_html = "<div class='og-actions-log'>No actions yet.</div>"
        st.markdown(log_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("WebRTC not connected yet. Click 'Start' in the video widget.")
