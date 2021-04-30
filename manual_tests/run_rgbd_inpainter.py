from remimi.utils.depth import get_depth
import torch
from remimi.inpaint.rgbd.mengli.inpainter import EdgeBasedRGBDInpainter, EdgeBasedRGBDInpainterOption
import cv2
from remimi.monodepth.dpt import DPTDepthEstimator

estimator = DPTDepthEstimator()

rgb_image = cv2.imread("data/rgbd_inpaint_test/00027.jpg")
mask_image = cv2.imread("data/rgbd_inpaint_test/00027.png")

disparity = estimator.estimate_depth_raw(rgb_image)
resized_disparity = torch.nn.functional.interpolate(
    disparity.unsqueeze(1),
    size=rgb_image.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze().cpu().numpy()

depth_image = get_depth(disparity)

inpainter = EdgeBasedRGBDInpainter()
inpainter.inpaint(rgb_image, depth_image, resized_disparity, mask_image)