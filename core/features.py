# core/features.py

import math
import numpy as np
from collections import deque


# ================================================================
# FEATURE EXTRACTOR (Landmarks → Relative, Angles, Finger States)
# ================================================================
class FeatureExtractor:
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # RELATIVE LANDMARKS (wrist-centered, normalized)
    # --------------------------------------------------------------
    @staticmethod
    def landmark_to_relative(landmarks):
        if not landmarks:
            return []

        wrist = landmarks[0]

        # normalize by wrist → middle fingertip distance
        hand_size = math.dist(
            (wrist.x, wrist.y),
            (landmarks[12].x, landmarks[12].y)
        ) + 1e-6

        rel = []
        for lm in landmarks:
            rel.append((
                (lm.x - wrist.x) / hand_size,
                (lm.y - wrist.y) / hand_size
            ))

        return rel

    # --------------------------------------------------------------
    # ANGLE CALCULATOR (3-point angle)
    # --------------------------------------------------------------
    @staticmethod
    def angle(a, b, c):
        """Angle between points a-b-c in radians."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.arccos(cosine)

    # --------------------------------------------------------------
    # FINGER STATES (stable angle-based detection)
    # --------------------------------------------------------------
    @staticmethod
    def finger_states(lm):
        """
        Returns 5 values (0/1): Thumb, Index, Middle, Ring, Pinky.
        Uses angle thresholds → extremely stable.
        """

        if lm is None or len(lm) < 21:
            return [0, 0, 0, 0, 0]

        pts = [(p.x, p.y) for p in lm]

        # Joint chains for fingers
        finger_joints = {
            "Thumb":  [2, 3, 4],
            "Index":  [5, 6, 8],
            "Middle": [9, 10, 12],
            "Ring":   [13, 14, 16],
            "Pinky":  [17, 18, 20],
        }

        states = []

        for name, (a, b, c) in finger_joints.items():
            ang = FeatureExtractor.angle(pts[a], pts[b], pts[c])

            # Stable threshold: 2.2 rad ≈ 126 degrees
            extended = ang > 2.2
            states.append(1 if extended else 0)

        return states

    # --------------------------------------------------------------
    # FINGER COUNT + NAMES
    # --------------------------------------------------------------
    @staticmethod
    def finger_count_and_names(finger_states):
        names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        fingers_up = [names[i] for i, v in enumerate(finger_states) if v == 1]
        return len(fingers_up), fingers_up

    # --------------------------------------------------------------
    # HAND BOUNDING BOX
    # --------------------------------------------------------------
    @staticmethod
    def hand_bounding_box(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    # --------------------------------------------------------------
    # HAND AREA (for zoom, push, pull)
    # --------------------------------------------------------------
    @staticmethod
    def hand_area(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))


# ================================================================
#     SMOOTHING FILTERS (Landmarks + finger states)
# ================================================================
class Smoother:
    """Smooths landmark positions using moving average."""
    def __init__(self, window_size=7):
        self.window = deque(maxlen=window_size)

    def smooth_points(self, points):
        self.window.append(points)
        avg = np.mean(self.window, axis=0)
        return [(float(x), float(y)) for x, y in avg]


class TemporalFingerStabilizer:
    """Smooths finger states using majority voting."""
    def __init__(self, history_size=7):
        self.history = deque(maxlen=history_size)

    def update(self, raw_states):
        self.history.append(raw_states)
        arr = np.array(self.history)
        stable = (np.mean(arr, axis=0) > 0.6).astype(int)
        return list(stable)
