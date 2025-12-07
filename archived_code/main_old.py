import cv2
import time
from collections import deque
from core.detection import HandDetector
from core.features import FeatureExtractor
from core.gestures import GestureEngine
from utils.async_helpers import VideoCaptureAsync
import pyautogui  # for future gesture → OS action mapping

# Initialize modules
gesture_engine = GestureEngine(buffer_len=5)
detector = HandDetector()
extractor = FeatureExtractor()
cam = VideoCaptureAsync(0)

# Buffer for recent gestures / actions
recent_actions = deque(maxlen=10)

# Emergency stop: make 'Fist' gesture as stop toggle
EMERGENCY_GESTURE = "Fist"
emergency_stop = False

prev_time = 0
frame_count = 0

def map_gesture_to_action(static_gesture, dynamic_gesture):
    """
    Example mapping of gestures to OS actions.
    Extend this for mouse/keyboard events using PyAutoGUI.
    """
    action = None
    if static_gesture == "Pinch":
        action = "Zoom In/Out (placeholder)"
        # pyautogui.hotkey('ctrl', '+')  # Example
    elif static_gesture == "Thumbs Up":
        action = "Play/Pause (placeholder)"
        # pyautogui.press('space')
    elif dynamic_gesture == "Swipe Left":
        action = "Previous Slide (placeholder)"
        # pyautogui.press('left')
    elif dynamic_gesture == "Swipe Right":
        action = "Next Slide (placeholder)"
        # pyautogui.press('right')
    return action

# ----------------------- Main Loop -----------------------
while True:
    frame = cam.read()
    if frame is None:
        continue

    # Hand detection
    results = detector.detect(frame)
    frame = detector.draw(frame, results)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Feature extraction
            rel_lms = extractor.landmark_to_relative(hand_landmarks.landmark)
            finger_states = extractor.finger_states(hand_landmarks.landmark)
            bbox = extractor.hand_bounding_box(hand_landmarks.landmark)

            # Compute centroid
            x_vals = [x for x,y in rel_lms]
            y_vals = [y for x,y in rel_lms]
            centroid = (sum(x_vals)/len(x_vals), sum(y_vals)/len(y_vals))
            gesture_engine.update_history(centroid)

            # Detect gestures
            static_gesture = gesture_engine.detect_static(rel_lms, finger_states)
            dynamic_gesture = gesture_engine.detect_dynamic()

            # Emergency stop toggle
            if static_gesture == EMERGENCY_GESTURE:
                emergency_stop = not emergency_stop
                recent_actions.append(f"Emergency Stop toggled: {emergency_stop}")
                time.sleep(0.5)  # debounce

            if not emergency_stop:
                # Map gestures to actions
                action = map_gesture_to_action(static_gesture, dynamic_gesture)
                if action:
                    recent_actions.append(action)
                    print("Action:", action)

            # Overlay info on frame
            cv2.putText(frame, f"Static: {static_gesture}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            if dynamic_gesture:
                cv2.putText(frame, f"Dynamic: {dynamic_gesture}", (10,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Draw bounding box
            x_min, y_min, x_max, y_max = bbox
            h, w, _ = frame.shape
            cv2.rectangle(frame,
                          (int(x_min*w), int(y_min*h)),
                          (int(x_max*w), int(y_max*h)),
                          (0,255,0), 2)

    # Overlay FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Overlay recent actions
    for i, act in enumerate(list(recent_actions)[-5:]):
        cv2.putText(frame, f"> {act}", (10,150+30*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("One-Gesture - Phase 4 Dashboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
