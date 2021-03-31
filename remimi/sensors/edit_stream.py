from remimi.edit.hifill.hifill import MaskEliminator
import cv2
import numpy as np

from remimi.segmentation.rgb_segmentation import SemanticSegmenter


class HumanEliminatedStream:
    def __init__(self, sensor):
        self.sensor = sensor
        self.semantic_segmentater = SemanticSegmenter()
        self.eliminator = MaskEliminator()

    def get_color(self):
        color = self.sensor.get_color()
        cv2.imshow("Original", color)
        color2 = self.semantic_segmentater.convert_to_semantic_image(color)
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)
        color2[color2 > 0] = 255

        kernel = np.ones((5,5),np.uint8)
        color2 = cv2.erode(color2,kernel,iterations = 1)
        cv2.imshow("Mask", color2)

        # import IPython; IPython.embed()

        return self.eliminator.eliminate_by_mask(color, cv2.cvtColor(color2, cv2.COLOR_GRAY2BGR))
