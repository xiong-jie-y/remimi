from remimi.inpaint.rgbd.ken3d.inpainter import JointRGBAndDepthInpainter, OccludingObjectsInpaintMethod
import cv2
import torch
from remimi.inpaint.rgbd.mengli.inpainter import (EdgeBasedRGBDInpainter,
                                                  EdgeBasedRGBDInpainterOption)
from remimi.monodepth.dpt import DPTDepthEstimator
import simple_parsing

def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--filename", required=True)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--mask-inpainter", default="joint_rgbd")
    EdgeBasedRGBDInpainter.add_arguments(parser)
    JointRGBAndDepthInpainter.add_arguments(parser)
    args = parser.parse_args()

    estimator = DPTDepthEstimator()
    rgb_image = cv2.imread(args.filename)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    if args.resize:
        rgb_image = cv2.resize(rgb_image, (1280, 720))
    depth_image_container = estimator.estimate_and_get_depth_image_container(rgb_image)

    if args.mask_inpainter == "edge_based":
        inpainter = EdgeBasedRGBDInpainter(debug=True)
        inpainter.inpaint_occluding_objects(rgb_image, depth_image_container)
    elif args.mask_inpainter == "joint_rgbd":
        inpainter = JointRGBAndDepthInpainter(option=JointRGBAndDepthInpainter.to_option(args))
        inpainter.inpaint_occluding_objects(rgb_image, depth_image_container)
    else:
        raise RuntimeError(f"No such methods {args.mask_inpainter}.")


if __name__ == "__main__":
    main()