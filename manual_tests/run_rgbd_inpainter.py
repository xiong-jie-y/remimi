import cv2
import torch
from remimi.inpaint.rgbd.mengli.inpainter import (EdgeBasedRGBDInpainter,
                                                  EdgeBasedRGBDInpainterOption)
from remimi.monodepth.dpt import DPTDepthEstimator

estimator = DPTDepthEstimator()

rgb_image = cv2.imread("data/rgbd_inpaint_test/00027.jpg")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
# mask_image = cv2.imread("data/rgbd_inpaint_test/00027.png")

depth_image_container = estimator.estimate_and_get_depth_image_container(rgb_image)

inpainter = EdgeBasedRGBDInpainter()
inpainter.inpaint(rgb_image, depth_image_container)
