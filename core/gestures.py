import math
from collections import deque

class GestureEngine:
    def __init__(self, buffer_len=5):
        self.history = deque(maxlen=buffer_len)

    @staticmethod
    def distance(lm1, lm2):
        return math.hypot(lm1[0]-lm2[0], lm1[1]-lm2[1])

    def detect_static(self, rel_landmarks, finger_states):
        """
        Detect static gestures from normalized landmarks & finger states
        Returns gesture label string
        """
        if all(not f for f in finger_states):
            return "Fist"
        if all(f for f in finger_states):
            return "Open"
        # Pinch: thumb + index close, others not extended
        if finger_states[0] and finger_states[1] and all(not f for f in finger_states[2:]):
            if self.distance(rel_landmarks[4], rel_landmarks[8]) < 0.15:
                return "Pinch"
        # Thumb up: only thumb extended
        if finger_states[0] and all(not f for f in finger_states[1:]):
            return "Thumbs Up"
        return "Unknown"

    def update_history(self, centroid):
        """Store hand centroid for dynamic gestures"""
        self.history.append(centroid)

    def detect_dynamic(self):
        """Detect simple swipe left/right based on centroid movement"""
        if len(self.history) < 2:
            return None
        dx = self.history[-1][0] - self.history[0][0]
        if dx > 0.3:
            return "Swipe Right"
        if dx < -0.3:
            return "Swipe Left"
        return None
