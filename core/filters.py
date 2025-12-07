import numpy as np
from collections import deque

class MovingAverageFilter:
    def __init__(self, window=5):
        self.buffer = deque(maxlen=window)

    def apply(self, pts):
        self.buffer.append(pts)
        arr = np.array(self.buffer)
        return np.mean(arr, axis=0).tolist()


class EMAFilter:
    """
    Exponential Moving Average Filter
    alpha = 0.0 → fully smooth but slow
    alpha = 1.0 → no smoothing
    For stability: alpha=0.35–0.45
    """
    def __init__(self, alpha=0.40):
        self.alpha = alpha
        self.prev = None

    def apply(self, pts):
        arr = np.array(pts, dtype=np.float32)
        if self.prev is None:
            self.prev = arr
            return pts

        filtered = self.alpha * arr + (1 - self.alpha) * self.prev
        self.prev = filtered
        return filtered.tolist()
