from threading import Thread
import queue

class VideoCaptureAsync:
    """
    Async wrapper for OpenCV VideoCapture
    Captures frames in a separate thread to avoid blocking
    """
    def __init__(self, src=0):
        import cv2
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Keep only the latest frame
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()
