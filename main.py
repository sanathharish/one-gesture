# main.py

import time
import threading
import math
from collections import deque

import cv2
import numpy as np
import pyautogui  # for OS mouse actions (scroll, clicks)

from PyQt5 import QtWidgets, QtCore

from core.detection import HandDetector
from core.features import FeatureExtractor, Smoother, TemporalFingerStabilizer
from core.gestures import GestureEngine
from ui.air_pointer import AirPointer

# ---------------------------------------------------------------------
# SHARED STATE BETWEEN VISION THREAD AND QT MAIN THREAD
# ---------------------------------------------------------------------
pointer_target_x = None
pointer_target_y = None

last_static_gesture = "None"
last_dynamic_gesture = "None"
last_finger_count = 0
last_finger_names = []


# ---------------------------------------------------------------------
# ONE EURO–STYLE FILTER (SIMPLE IMPLEMENTATION)
# ---------------------------------------------------------------------
class OneEuro1D:
    """
    Simple One-Euro-like filter:
    - more smoothing when speed is low
    - more responsive when speed is high
    """

    def __init__(self, alpha_min=0.15, alpha_max=0.7, speed_scale=80.0):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.speed_scale = speed_scale
        self.prev = None

    def update(self, value, speed):
        if self.prev is None:
            self.prev = value
            return value

        # Normalize speed to [0, 1] range for blending.
        t = max(0.0, min(1.0, speed * self.speed_scale))
        alpha = self.alpha_min + t * (self.alpha_max - self.alpha_min)

        filtered = self.prev + alpha * (value - self.prev)
        self.prev = filtered
        return filtered


# ---------------------------------------------------------------------
# HUD DRAWING (same layout as before)
# ---------------------------------------------------------------------
def draw_hud(frame, fps, static_gesture, dynamic_gesture,
             finger_count, finger_names, recent_actions):

    h, w, _ = frame.shape

    panel_h = int(h * 0.18)
    y1 = h - panel_h
    y2 = h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (10, 25, 40), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    cv2.line(frame, (0, y1), (w, y1), (0, 255, 255), 2)

    col_w = w // 3
    x_left = int(col_w * 0.05)
    x_mid = col_w + int(col_w * 0.05)
    x_right = 2 * col_w + int(col_w * 0.05)

    y_base = y1 + int(panel_h * 0.22)
    dy = int(panel_h * 0.20)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Left column
    cv2.putText(frame, f"FPS: {int(fps)}",
                (x_left, y_base),
                font, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, "Static:", (x_left, y_base + dy),
                font, 0.6, (200, 255, 255), 1)
    cv2.putText(frame, static_gesture,
                (x_left + 120, y_base + dy),
                font, 0.75, (255, 255, 255), 2)

    cv2.putText(frame, "Dynamic:", (x_left, y_base + 2 * dy),
                font, 0.6, (200, 255, 255), 1)
    cv2.putText(frame, dynamic_gesture,
                (x_left + 120, y_base + 2 * dy),
                font, 0.75, (255, 255, 255), 2)

    # Middle column
    cv2.putText(frame, f"Fingers: {finger_count}",
                (x_mid, y_base),
                font, 0.8, (0, 255, 255), 2)

    finger_text = "None" if not finger_names else " | ".join(finger_names)
    cv2.putText(frame, finger_text,
                (x_mid, y_base + dy),
                font, 0.6, (230, 255, 255), 1)

    # Right column
    cv2.putText(frame, "Recent Actions:",
                (x_right, y_base),
                font, 0.6, (180, 255, 255), 1)

    for i, act in enumerate(list(recent_actions)[-4:]):
        cv2.putText(frame, act,
                    (x_right, y_base + (i + 1) * dy),
                    font, 0.55, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------
# VISION / GESTURE THREAD (OpenCV + Mediapipe + HUD + OS actions)
# ---------------------------------------------------------------------
def vision_loop(virtual_width, virtual_height, offset_x, offset_y):
    global pointer_target_x, pointer_target_y
    global last_static_gesture, last_dynamic_gesture
    global last_finger_count, last_finger_names

    detector = HandDetector()
    extractor = FeatureExtractor()
    gesture_engine = GestureEngine(buffer_len=20)
    smoother = Smoother(window_size=6)
    finger_stabilizer = TemporalFingerStabilizer(history_size=7)

    recent_actions = deque(maxlen=10)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_time = 0.0
    window_name = "One-Gesture (Fullscreen HUD)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name,
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Pointer state (absolute in virtual desktop coords)
    pointer_x = offset_x + virtual_width / 2
    pointer_y = offset_y + virtual_height / 2

    # One-Euro filters for fingertip
    fx_filter = OneEuro1D()
    fy_filter = OneEuro1D()

    prev_raw_fx = None
    prev_raw_fy = None

    prev_fx_filtered = None
    prev_fy_filtered = None

    # Movement config (tweakable)
    DEAD_ZONE = 0.0015        # hand jitter threshold in normalized units
    BASE_GAIN = 0.18          # base pointer speed
    ACCEL_FACTOR = 2.2        # how strongly speed increases gain
    MAX_SPEED_NORM = 0.06     # clamp speed influence
    EDGE_SNAP_PX = 12         # snap margin in pixels near edges

    min_x = offset_x
    max_x = offset_x + virtual_width - 1
    min_y = offset_y
    max_y = offset_y + virtual_height - 1

    # Pinch/drag state
    drag_active = False
    pinch_prev = False
    pinch_start_time = 0.0
    right_click_cooldown = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        results = detector.detect(frame)
        frame = detector.draw(frame, results)

        static_gesture = "None"
        dynamic_gesture = "None"
        finger_names = []
        finger_count = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # --- Relative landmarks for gestures ---
            raw_lms = extractor.landmark_to_relative(hand.landmark)
            rel_lms = smoother.smooth_points(raw_lms)

            # --- Finger states (stable) ---
            raw_states = extractor.finger_states(hand.landmark)
            finger_states = finger_stabilizer.update(raw_states)

            thumb, index, middle, ring, pinky = finger_states
            finger_count, finger_names = extractor.finger_count_and_names(
                finger_states
            )

            # --- Gesture detection (unchanged) ---
            cx = sum(p[0] for p in rel_lms) / len(rel_lms)
            cy = sum(p[1] for p in rel_lms) / len(rel_lms)
            gesture_engine.update_history((cx, cy))

            sg = gesture_engine.detect_static(rel_lms, finger_states)
            static_gesture = gesture_engine.filter_static(sg)
            dynamic_gesture = gesture_engine.detect_dynamic()

            if static_gesture != "None":
                recent_actions.append(static_gesture)
            if dynamic_gesture != "None":
                recent_actions.append(dynamic_gesture)

            # ---------------- MODE LOGIC ---------------- #
            up_count = sum(finger_states)
            mode = "move"

            # Open hand → idle (freeze pointer)
            if up_count == 5:
                mode = "idle"
            # two fingers (index + middle) → scroll mode
            elif index == 1 and middle == 1 and ring == 0 and pinky == 0:
                mode = "scroll"
            # only index up → move
            elif index == 1 and thumb == 0 and middle == 0 and ring == 0 and pinky == 0:
                mode = "move"
            else:
                # default: allow move
                mode = "move"

            # ---------------- PINCH / CLICK / DRAG / RIGHT CLICK ---------------- #
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]

            dist_pinch = math.dist(
                (thumb_tip.x, thumb_tip.y),
                (index_tip.x, index_tip.y),
            )
            dist_right = math.dist(
                (thumb_tip.x, thumb_tip.y),
                (middle_tip.x, middle_tip.y),
            )

            PINCH_THRESH = 0.035
            RIGHT_THRESH = 0.04

            is_pinch = dist_pinch < PINCH_THRESH
            is_right_pinch = dist_right < RIGHT_THRESH

            now_t = time.time()

            # Right-click with cooldown
            if is_right_pinch and (now_t - right_click_cooldown) > 0.4:
                pyautogui.click(button="right")
                right_click_cooldown = now_t
                recent_actions.append("Right Click")

            # Left pinching → click / drag
            if is_pinch and not pinch_prev:
                # Pinch began
                pinch_start_time = now_t
                drag_active = True
                pyautogui.mouseDown(button="left")
                recent_actions.append("Drag Start")

            elif not is_pinch and pinch_prev:
                # Pinch ended
                pinch_duration = now_t - pinch_start_time
                pyautogui.mouseUp(button="left")
                drag_active = False

                if pinch_duration < 0.25:
                    # Treat as simple click
                    pyautogui.click(button="left")
                    recent_actions.append("Left Click")
                else:
                    recent_actions.append("Drag End")

            pinch_prev = is_pinch

            # ---------------- RELATIVE POINTER MOVEMENT ---------------- #
            idx_tip = hand.landmark[8]
            raw_fx = idx_tip.x
            raw_fy = idx_tip.y

            if prev_raw_fx is None:
                prev_raw_fx = raw_fx
                prev_raw_fy = raw_fy
                prev_fx_filtered = raw_fx
                prev_fy_filtered = raw_fy

            # Raw deltas (for speed & scroll)
            dx_raw = raw_fx - prev_raw_fx
            dy_raw = raw_fy - prev_raw_fy
            prev_raw_fx = raw_fx
            prev_raw_fy = raw_fy

            speed_norm = math.sqrt(dx_raw * dx_raw + dy_raw * dy_raw)

            # Filter fingertip with One-Euro
            fx = fx_filter.update(raw_fx, speed_norm)
            fy = fy_filter.update(raw_fy, speed_norm)

            # Filtered deltas for pointer movement
            dx = fx - prev_fx_filtered
            dy = fy - prev_fy_filtered
            prev_fx_filtered = fx
            prev_fy_filtered = fy

            # Dead zone to stop micro jitter
            if speed_norm < DEAD_ZONE or mode == "idle":
                dx = 0.0
                dy = 0.0

            if mode == "move":
                # Adaptive gain (pointer acceleration)
                s = min(speed_norm, MAX_SPEED_NORM)
                gain = BASE_GAIN + ACCEL_FACTOR * (s / MAX_SPEED_NORM)

                delta_x = dx * gain * virtual_width
                delta_y = dy * gain * virtual_height

                pointer_x += delta_x
                pointer_y += delta_y

                # Clamp to desktop bounds
                pointer_x = float(np.clip(pointer_x, min_x, max_x))
                pointer_y = float(np.clip(pointer_y, min_y, max_y))

                # Edge snap
                if pointer_x - min_x < EDGE_SNAP_PX:
                    pointer_x = min_x
                elif max_x - pointer_x < EDGE_SNAP_PX:
                    pointer_x = max_x

                if pointer_y - min_y < EDGE_SNAP_PX:
                    pointer_y = min_y
                elif max_y - pointer_y < EDGE_SNAP_PX:
                    pointer_y = max_y

                pointer_target_x = pointer_x
                pointer_target_y = pointer_y

            elif mode == "scroll":
                # Vertical scroll using dy_raw (raw movement more responsive)
                SCROLL_SCALE = 1500.0
                scroll_amount = dy_raw * SCROLL_SCALE

                if abs(scroll_amount) > 0.3:
                    # Negative scroll moves down in many systems, adjust as needed
                    pyautogui.scroll(int(-scroll_amount))
                    recent_actions.append(
                        "Scroll Up" if scroll_amount < 0 else "Scroll Down"
                    )

            elif mode == "idle":
                # Do nothing, pointer stays
                pass

        # Save debug state
        last_static_gesture = static_gesture
        last_dynamic_gesture = dynamic_gesture
        last_finger_count = finger_count
        last_finger_names = finger_names

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now

        frame = draw_hud(
            frame,
            fps=fps,
            static_gesture=static_gesture,
            dynamic_gesture=dynamic_gesture,
            finger_count=finger_count,
            finger_names=finger_names,
            recent_actions=recent_actions,
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# QT MAIN (AIR POINTER OVERLAY + MULTI-MONITOR DETECTION)
# ---------------------------------------------------------------------
def main():
    global pointer_target_x, pointer_target_y

    app = QtWidgets.QApplication([])

    # Detect all screens and compute virtual desktop rect
    screens = app.screens()
    xs, ys, x2s, y2s = [], [], [], []

    print("[INFO] Detected screens:")
    for i, s in enumerate(screens):
        g = s.geometry()
        sx, sy, sw, sh = g.x(), g.y(), g.width(), g.height()
        print(f"  Screen {i}: x={sx}, y={sy}, w={sw}, h={sh}")
        xs.append(sx)
        ys.append(sy)
        x2s.append(sx + sw)
        y2s.append(sy + sh)

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2s)
    y_max = max(y2s)

    virtual_width = x_max - x_min
    virtual_height = y_max - y_min

    print(f"[INFO] Virtual desktop rect: x={x_min}, y={y_min}, "
          f"w={virtual_width}, h={virtual_height}")

    offset_x = x_min
    offset_y = y_min

    # Neon overlay pointer
    air_pointer = AirPointer(size=40)

    # Start vision loop in background thread
    t = threading.Thread(
        target=vision_loop,
        args=(virtual_width, virtual_height, offset_x, offset_y),
        daemon=True
    )
    t.start()

    # Timer updates overlay from shared pointer_target_x/y
    def update_overlay():
        if pointer_target_x is not None and pointer_target_y is not None:
            air_pointer.update_position(pointer_target_x, pointer_target_y)

    timer = QtCore.QTimer()
    timer.timeout.connect(update_overlay)
    timer.start(16)  # ~60 FPS

    app.exec_()


if __name__ == "__main__":
    main()
