import cv2

class SimpleWebcamera:
    def __init__(self, cam_id):
        self.cap = cv2.VideoCapture(cam_id)

    def get_color(self):
        flag, frame = self.cap.read()

        return frame