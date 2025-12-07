import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands, min_detection_confidence=0.7, min_tracking_confidence=0.6)
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):
        """Returm processed frame and landmark data."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results
    
    def draw(self, frame, results):
        """Render detected landmarks on the frame."""
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
        return frame