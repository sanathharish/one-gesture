import math
from collections import deque
import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def landmark_to_relative(landmarks):
        """
        Convert absolute MediaPipe landmarks to relative coordinates
        relative to the wrist (landmark 0) and normalized by hand size.
        """
        if not landmarks:
            return []

        wrist = landmarks[0]
        rel_landmarks = []

        # Compute hand size as distance from wrist to middle finger tip
        hand_size = math.dist((wrist.x, wrist.y), 
                              (landmarks[12].x, landmarks[12].y)) + 1e-6

        for lm in landmarks:
            rel_x = (lm.x - wrist.x) / hand_size
            rel_y = (lm.y - wrist.y) / hand_size
            rel_landmarks.append((rel_x, rel_y))
        return rel_landmarks

    @staticmethod
    def finger_states(landmarks):
        """
        Determine if fingers are open or closed (simple heuristic).
        Returns list of 5 bools: [thumb, index, middle, ring, pinky]
        """
        if not landmarks:
            return [False]*5

        states = []
        # Thumb: tip vs MCP x coordinate (simple left/right check)
        states.append(landmarks[4].x > landmarks[3].x)
        # Other fingers: tip y < PIP y => extended
        fingers = [(8,6), (12,10), (16,14), (20,18)]
        for tip, pip in fingers:
            states.append(landmarks[tip].y < landmarks[pip].y)
        return states

    @staticmethod
    def hand_bounding_box(landmarks):
        """
        Compute axis-aligned bounding box for the hand.
        Returns (x_min, y_min, x_max, y_max)
        """
        if not landmarks:
            return 0,0,0,0

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

class Smoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def smooth_points(self, points):
        """
        points = list of (x,y) tuples
        """
        self.buffer.append(points)
        avg = np.mean(self.buffer, axis=0)
        return [(float(p[0]), float(p[1])) for p in avg]

