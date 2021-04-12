from remimi.sensors import StreamFinished
import cv2

class MultipleImageStream:
    def __init__(self, image_paths):
        self.images = [cv2.imread(image_path) for image_path in image_paths]
        self.next_image = iter(self.images)
    
    def get_color(self):
        try:
            return next(self.next_image)
        except StopIteration:
            raise StreamFinished