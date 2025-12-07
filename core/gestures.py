# core/gestures.py

import math
from collections import deque

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class GestureEngine:
    def __init__(self, buffer_len=20):
        # Buffer of centroid points for swipe recognition
        self.centroid_history = deque(maxlen=buffer_len)

        # Buffer for static gesture smoothing
        self.static_history = deque(maxlen=12)

    # ============================================================
    # MOVEMENT HISTORY (for swipe gestures)
    # ============================================================
    def update_history(self, centroid):
        """
        centroid = (cx, cy)
        used to detect directional motion
        """
        self.centroid_history.append(centroid)

    # ============================================================
    # STATIC GESTURE DETECTION (raw, per-frame)
    # ============================================================
    def detect_static(self, rel_lms, finger_states):
        """
        finger_states: [thumb, index, middle, ring, pinky] (1=open, 0=closed)
        """

        if not finger_states or len(finger_states) != 5:
            return "None"

        up_count = sum(finger_states)
        thumb, index, middle, ring, pinky = finger_states

        # -------------------------
        # 0 — Hand states
        # -------------------------
        if up_count == 0:
            return "Closed Hand"

        if up_count == 5:
            return "Open Hand"

        # -------------------------
        # 1 — Single finger gestures
        # -------------------------
        if up_count == 1:
            idx = finger_states.index(1)
            return f"{FINGER_NAMES[idx]} Up"

        # -------------------------
        # 2 — Two finger combinations
        # -------------------------
        if up_count == 2:
            idxs = [i for i, v in enumerate(finger_states) if v == 1]
            return f"{FINGER_NAMES[idxs[0]]} + {FINGER_NAMES[idxs[1]]}"

        # -------------------------
        # 3 — Point
        # -------------------------
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return "Point"

        # -------------------------
        # 4 — Thumbs Up
        # -------------------------
        if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
            return "Thumbs Up"

        # -------------------------
        # 5 — OK (thumb touching index)
        # -------------------------
        thumb_tip = rel_lms[4]
        index_tip = rel_lms[8]
        dist = math.dist(thumb_tip, index_tip)

        if dist < 0.05:
            return "OK"

        # -------------------------
        # Default fallback
        # -------------------------
        return f"{up_count} Fingers Up"

    # ============================================================
    # STATIC FILTERING (zero flicker)
    # ============================================================
    def filter_static(self, gesture):
        """
        Stabilizes static gestures.
        Keeps a rolling history and only outputs a gesture
        when it appears in >70% of the last N frames.
        """

        self.static_history.append(gesture)

        # Wait until enough samples collected
        if len(self.static_history) < 10:
            return "None"

        # Count frequency of each gesture
        freq = {}
        for g in self.static_history:
            freq[g] = freq.get(g, 0) + 1

        # Most frequent gesture
        best = max(freq, key=freq.get)

        # Must be dominant > 70%
        if freq[best] / len(self.static_history) > 0.7:
            if best != "None":
                return best

        return "None"

    # ============================================================
    # DYNAMIC GESTURES: Swipe detection (robust)
    # ============================================================
    def detect_dynamic(self):
        """
        Swipe detection based on centroid displacement + velocity.
        """

        if len(self.centroid_history) < 8:
            return "None"

        # Extract motion arrays
        xs = [p[0] for p in self.centroid_history]
        ys = [p[1] for p in self.centroid_history]

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]

        # Short-term velocity (helps reduce noise)
        vel_x = xs[-1] - xs[-3]
        vel_y = ys[-1] - ys[-3]

        # Thresholds (tuned for CPU + webcam)
        DISP_THRESH = 0.020     # overall travel required
        VEL_THRESH = 0.012      # short-term punch required

        # -------------------------
        # HORIZONTAL SWIPES
        # -------------------------
        if abs(vel_x) > abs(vel_y) and abs(dx) > DISP_THRESH and abs(vel_x) > VEL_THRESH:
            return "Swipe Right" if vel_x > 0 else "Swipe Left"

        # -------------------------
        # VERTICAL SWIPES
        # -------------------------
        if abs(vel_y) > abs(dx) and abs(dy) > DISP_THRESH and abs(vel_y) > VEL_THRESH:
            return "Swipe Down" if vel_y > 0 else "Swipe Up"

        return "None"
