import cv2
import threading
import time

class RTSPClient:
    def __init__(self, url, cam_id):
        self.url = url
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.drop_count = 0

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.drop_count += 1
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    cam1 = RTSPClient("data/videos/traffic1.mp4", "cam1").start()
    cam2 = RTSPClient("rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov", "cam2").start()

    while True:
        f1 = cam1.read()
        f2 = cam2.read()
        if f1 is not None:
            cv2.imshow("Cam1", f1)
        if f2 is not None:
            cv2.imshow("Cam2", f2)
        if cv2.waitKey(1) == 27:
            break

    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()
