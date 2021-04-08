import os
import glob
from remimi.monodepth.ken3d.depthestim import Ken3DDepthEstimator
from remimi.monodepth.dpt import DPTDepthEstimator
# from remimi.monodepth.adabin import InferenceHelper
from remimi.segmentation.rgb_segmentation import SemanticSegmenter
import torch
import cv2
import argparse
import numpy as np
import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from util.misc import visualize_attention

import enum
class ImageType(enum.Enum):
    RGB = "rgb"
    BGR = "bgr"

import matplotlib.pyplot as plt

def show_image_blocking(image, **imshow_arg):
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(image.shape[1], image.shape[0])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto', **imshow_arg)
    plt.show()

class DPTPaseudoDepthCamera:
    def __init__(self, sensor, model_name: str, debug=False, output_type=ImageType.RGB, boundary_depth_removal=False):
        if model_name == "dpt":
            self.depth_estimator = DPTDepthEstimator(debug)
        elif model_name == "ken3d":
            self.depth_estimator = Ken3DDepthEstimator(debug=debug)
        # self.depth_estimator = InferenceHelper()

        self.sensor = sensor
        self.output_type = output_type
        self.boundary_depth_removal = boundary_depth_removal
        self.semantic_segmentater = SemanticSegmenter()

    def get_color_and_depth(self):
        color = self.sensor.get_color()
        # color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = util.io.read_image(img_name)
        depth = self.depth_estimator.estimate_depth(color)

        # import IPython; IPython.embed()

        if self.boundary_depth_removal:
            color2 = self.semantic_segmentater.convert_to_semantic_image(color)
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            cv2.imshow("semantic", color2)
            # color2 = np.copy(smimg)
            # show_image_blocking(color_rgb)
            color2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)
            color2[color2 > 0] = 255
            # show_image_blocking(color2, cmap = "gray")
            canny_img = cv2.Canny(color2, 50, 110)
            kernel = np.ones((5,5),np.uint8)
            canny_img = cv2.dilate(canny_img,kernel,iterations = 1)

            cv2.imshow("test2222", canny_img)

            depth[canny_img == 255] = 0

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if self.output_type == ImageType.BGR:
            return color, depth
        elif self.output_type == ImageType.RGB:
            return color, depth
        else:
            raise RuntimeError(f"no such image type {self.output_type}.")