import os
from os.path import join
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

        color2 = self.semantic_segmentater.get_mask(color, ["person"])

        kernel = np.ones((5,5),np.uint8)
        color2 = cv2.erode(color2,kernel,iterations = 3)
        cv2.imshow("Mask", color2)

        return self.eliminator.eliminate_by_mask(color, cv2.cvtColor(color2, cv2.COLOR_GRAY2BGR))

class SaveMaskAndFrameSink:
    def __init__(self, stream, output_root, class_names):
        self.stream = stream
        os.makedirs(join(output_root, "masks"), exist_ok=True)
        os.makedirs(join(output_root, "frames"), exist_ok=True)
        self.frame_count = 0
        
        self.semantic_segmentater = SemanticSegmenter()
        self.class_names = class_names
        self.output_root = output_root

    def process(self, show=False):
        filename = str(self.frame_count).zfill(5)
        color = self.stream.get_color()
        if show:
            cv2.imshow("Original", color)
        cv2.imwrite(join(self.output_root, "frames/{}.jpg".format(filename)), color)

        color2 = self.semantic_segmentater.get_mask(color, self.class_names)

        white_mask = np.zeros(color2.shape, dtype=np.uint8)
        white_mask[color2 == 0] = 255
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2RGB)
        if show:
            cv2.imshow("Mask", white_mask)
        cv2.imwrite(join(self.output_root, "masks/{}.png".format(filename)), white_mask)

        self.frame_count += 1
