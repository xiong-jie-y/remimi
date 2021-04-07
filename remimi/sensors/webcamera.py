from remimi.sensors import StreamFinished
import cv2

class SimpleWebcamera:
    def __init__(self, cam_id, fps=None):
        self.cap = cv2.VideoCapture(cam_id)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_color(self):
        flag, frame = self.cap.read()

        if not flag:
            raise StreamFinished()

        return frame

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)