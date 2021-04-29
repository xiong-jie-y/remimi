import os
from os.path import join
from remimi.detection.instance_segmentation import InstanceSegmenter
from remimi.edit.hifill.hifill import MaskEliminator
import cv2
import numpy as np

from remimi.segmentation.rgb_segmentation import SemanticSegmenter


class HumanEliminatedStream:
    def __init__(self, sensor, margin=1):
        self.sensor = sensor
        # self.semantic_segmentater = SemanticSegmenter()
        self.semantic_segmentater = InstanceSegmenter()
        self.eliminator = MaskEliminator()
        self.margin = margin

    def get_color(self):
        color = self.sensor.get_color()
        cv2.imshow("Original", color)

        color2 = self.semantic_segmentater.get_mask(color, ["person"])

        kernel = np.ones((5,5),np.uint8)
        color2 = cv2.erode(color2,kernel,iterations = self.margin)
        cv2.imshow("Mask", color2)

        return self.eliminator.eliminate_by_mask(color, cv2.cvtColor(color2, cv2.COLOR_GRAY2BGR))


class CustomizableHumanEliminatedStream:
    def __init__(self, sensor, mask_stream, margin=1):
        self.sensor = sensor
        self.eliminator = MaskEliminator()
        self.mask_stream = mask_stream
        self.margin = margin

    def get_color(self):
        color = self.sensor.get_color()
        cv2.imshow("Original", color)

        color2 = self.mask_stream.get_color()

        color2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)
        black_pass_mask = np.zeros(color2.shape, dtype=np.uint8)
        black_pass_mask[color2 > 125] = 0
        black_pass_mask[color2 < 125] = 255

        cv2.imshow("inpaint mask", black_pass_mask)

        # import IPython; IPython.embed()

        return self.eliminator.eliminate_by_mask(color, cv2.cvtColor(black_pass_mask, cv2.COLOR_GRAY2BGR))


OUTPUT_SIZE = (819, 455)

class SaveMaskAndFrameSink:
    def __init__(self, stream, output_root, class_names, margin):
        self.stream = stream
        os.makedirs(join(output_root, "masks"), exist_ok=True)
        os.makedirs(join(output_root, "frames"), exist_ok=True)
        os.makedirs(join(output_root, "originals"), exist_ok=True)
        self.frame_count = 0
        
        # self.semantic_segmentater = SemanticSegmenter()
        self.semantic_segmentater = InstanceSegmenter()
        self.class_names = class_names
        self.output_root = output_root
        self.margin = margin

    def process(self, show=False):
        filename = str(self.frame_count).zfill(5)
        color = self.stream.get_color()
        color_small = cv2.resize(color, OUTPUT_SIZE)
        color_medium = cv2.resize(color, (1280, 720))
        # color_upper = cv2.resize(color, (1280, 720))
        cv2.imwrite(join(self.output_root, "originals/{}.jpg".format(filename)), color, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(join(self.output_root, "frames/{}.jpg".format(filename)), color_small, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(join(self.output_root, "originals/medium{}.png".format(filename)), color_medium)
        cv2.imshow("Original Image", color_small)

        # color = color_upper

        color_yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
        color_yuv[:,:,0] = cv2.equalizeHist(color_yuv[:,:,0])
        color = cv2.cvtColor(color_yuv, cv2.COLOR_YUV2BGR)

        # color = cv2.resize(color, (1280, 720))
        color2 = self.semantic_segmentater.get_mask(color, self.class_names)

        if self.margin < 0:
            kernel = np.ones((5,5),np.uint8)
            color2 = cv2.dilate(color2,kernel,iterations = self.margin)
        elif self.margin > 0:
            kernel = np.ones((5,5),np.uint8)
            color2 = cv2.erode(color2,kernel,iterations = self.margin)

        white_mask = np.zeros(color2.shape, dtype=np.uint8)
        white_mask[color2 == 0] = 255
        white_mask = cv2.resize(white_mask, OUTPUT_SIZE)
        white_mask[white_mask > 128] = 255
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2RGB)

        if show:
            color_image_bgr = cv2.addWeighted(color, 0.5, cv2.cvtColor(color2, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow("Original", color_image_bgr)

        if show:
            cv2.imshow("Mask", white_mask)
        cv2.imwrite(join(self.output_root, "masks/{}.png".format(filename)), white_mask)

        self.frame_count += 1
