# core/features.py

import math
from collections import deque
import numpy as np


class FeatureExtractor:
    def __init__(self):
        pass

    # ----------------------------------------------------------------------
    # RELATIVE LANDMARKS (normalized & wrist-centered)
    # ----------------------------------------------------------------------
    @staticmethod
    def landmark_to_relative(landmarks):
        if not landmarks:
            return []

        wrist = landmarks[0]

        # Hand size for scaling
        hand_size = math.dist(
            (wrist.x, wrist.y),
            (landmarks[12].x, landmarks[12].y)
        ) + 1e-6

        rel_landmarks = []
        for lm in landmarks:
            rel_landmarks.append((
                (lm.x - wrist.x) / hand_size,
                (lm.y - wrist.y) / hand_size
            ))

        return rel_landmarks

    # ----------------------------------------------------------------------
    # FINGER STATES (True = finger extended)
    # ----------------------------------------------------------------------
    @staticmethod
    def finger_states(landmarks):
        if not landmarks:
            return [0, 0, 0, 0, 0]

        states = []

        # Thumb: compare tip.x vs MCP.x depending on left/right hand
        # More reliable thumb rule:
        thumb_tip = landmarks[4]
        thumb_ip  = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_cmc = landmarks[1]

        # Thumb direction vector
        thumb_dir = thumb_tip.x - thumb_ip.x

        states.append(1 if thumb_dir > 0 else 0)

        # Other fingers: tip y < pip y -> extended
        fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]

        for tip_idx, pip_idx in fingers:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]

            states.append(1 if tip.y < pip.y else 0)

        return states

    # ----------------------------------------------------------------------
    # FINGER COUNT + NAMES
    # ----------------------------------------------------------------------
    @staticmethod
    def finger_count_and_names(finger_states):
        """
        finger_states = [thumb, index, middle, ring, pinky]
        returns (count, ["Index", "Middle", ...])
        """

        names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        fingers_up = [
            names[i] for i, state in enumerate(finger_states) if state == 1
        ]

        return len(fingers_up), fingers_up

    # ----------------------------------------------------------------------
    # BOUNDING BOX
    # ----------------------------------------------------------------------
    @staticmethod
    def hand_bounding_box(landmarks):
        if not landmarks:
            return (0, 0, 0, 0)

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        return min(xs), min(ys), max(xs), max(ys)


# ----------------------------------------------------------------------
# SMOOTHER (Used for stabilizing landmarks)
# ----------------------------------------------------------------------
class Smoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def smooth_points(self, points):
        """
        points = list of (x, y)
        Returns smoothed list of same shape
        """
        self.window.append(points)
        avg = np.mean(self.window, axis=0)
        return [(float(x), float(y)) for x, y in avg]

# ----------------------------------------------------------------------
# TEMPORAL FINGER STABILIZER
# ----------------------------------------------------------------------
# Finger State Majority Voting (HUGE FIX)
class TemporalFingerStabilizer:
    """
    Smooths finger states over time using majority voting.
    """
    def __init__(self, history_size=7):
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

    def update(self, states):
        """
        states = [thumb,index,middle,ring,pinky] each frame.
        """
        self.history.append(states)

        # Convert history into stable state using majority rule
        arr = np.array(self.history)
        stable = (np.mean(arr, axis=0) > 0.6).astype(int)

        return list(stable)
