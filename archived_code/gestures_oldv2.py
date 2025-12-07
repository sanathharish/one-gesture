# core/gestures.py

from collections import deque
import math

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class GestureEngine:
    def __init__(self, buffer_len=20):
        self.centroid_history = deque(maxlen=buffer_len)
        self.static_history = deque(maxlen=10)
        self.dynamic_history = deque(maxlen=10)

        # cooldown frames to prevent spamming gestures
        self.cooldown_static = 0
        self.cooldown_dynamic = 0

    # ---------------------------------------------------------
    # UPDATE MOTION HISTORY
    # ---------------------------------------------------------
    def update_history(self, centroid):
        self.centroid_history.append(centroid)

    # ---------------------------------------------------------
    # STATIC GESTURES
    # ---------------------------------------------------------
    def detect_static(self, rel_lms, finger_states):
        if not finger_states or len(finger_states) != 5:
            return "None"

        up = sum(finger_states)
        thumb, index, middle, ring, pinky = finger_states

        if up == 0:
            return "Closed Hand"
        if up == 5:
            return "Open Hand"

        if up == 1:
            i = finger_states.index(1)
            return f"{FINGER_NAMES[i]} Up"

        if up == 2:
            idxs = [i for i, v in enumerate(finger_states) if v == 1]
            return f"{FINGER_NAMES[idxs[0]]} + {FINGER_NAMES[idxs[1]]}"

        # OK
        if math.dist(rel_lms[4], rel_lms[8]) < 0.05:
            return "OK"

        return f"{up} Fingers Up"

    # Debounce static
    def filter_static(self, gesture):
        if self.cooldown_static > 0:
            self.cooldown_static -= 1
            return "None"

        self.static_history.append(gesture)
        if self.static_history.count(gesture) >= 6:
            self.cooldown_static = 10
            return gesture

        return "None"

    # ---------------------------------------------------------
    # DYNAMIC GESTURES (Swipe)
    # ---------------------------------------------------------
    def detect_dynamic(self):
        if len(self.centroid_history) < 8:
            return "None"

        xs = [p[0] for p in self.centroid_history]
        ys = [p[1] for p in self.centroid_history]

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]

        vel_x = xs[-1] - xs[-3]
        vel_y = ys[-1] - ys[-3]

        DISP_THRESH = 0.08
        VEL_THRESH = 0.05
        DIR_CONSISTENCY = 0.8  # % frames where direction matches final direction

        # direction sign history
        x_signs = [1 if xs[i] - xs[i-1] > 0 else -1 for i in range(1, len(xs))]
        y_signs = [1 if ys[i] - ys[i-1] > 0 else -1 for i in range(1, len(ys))]

        # majority vote (intent detection)
        x_consistent = x_signs.count(1) / len(x_signs) if dx > 0 else x_signs.count(-1) / len(x_signs)
        y_consistent = y_signs.count(1) / len(y_signs) if dy > 0 else y_signs.count(-1) / len(y_signs)

        # ------------------------------------------------------------------
        # COOL DOWN
        # ------------------------------------------------------------------
        if self.cooldown_dynamic > 0:
            self.cooldown_dynamic -= 1
            return "None"

        # ------------------------------------------------------------------
        # SWIPE HORIZONTAL
        # ------------------------------------------------------------------
        if abs(dx) > abs(dy) and abs(dx) > DISP_THRESH and abs(vel_x) > VEL_THRESH and x_consistent > DIR_CONSISTENCY:
            self.cooldown_dynamic = 10
            return "Swipe Right" if dx > 0 else "Swipe Left"

        # ------------------------------------------------------------------
        # SWIPE VERTICAL
        # ------------------------------------------------------------------
        if abs(dy) > abs(dx) and abs(dy) > DISP_THRESH and abs(vel_y) > VEL_THRESH and y_consistent > DIR_CONSISTENCY:
            self.cooldown_dynamic = 10
            return "Swipe Down" if dy > 0 else "Swipe Up"

        return "None"
