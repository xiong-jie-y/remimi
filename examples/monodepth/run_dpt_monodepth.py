"""Compute depth maps for images in the input folder.
"""
import os
import glob
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

class DPTPaseudoDepthCamera:
    def __init__(self, model_path, sensor, model_type="dpt_hybrid", optimize=True, debug=False):
        print("initialize")

        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % device)

        net_w = net_h = 384

        # load network
        if model_type == "dpt_large":  # DPT-Large
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=debug,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=debug,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "midas_v21":  # Convolutional model
            model = MidasNet_large(model_path, non_negative=True)
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|midas_v21]"

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize == True and device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(device)

        self.model = model
        self.transform = transform
        self.device = device
        self.optimize = optimize

        # self.cap = cv2.VideoCapture(camera_id)
        self.debug = debug

        self.sensor = sensor

    def get_next_frame(self):
        img = self.sensor.get_color()
        # img = util.io.read_image(img_name)
        img_input = self.transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if self.debug:
                visualize_attention(sample, self.model, prediction, args.model_type)

        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = get_inverse_map(prediction.astype(np.uint16))

        return color, depth

import open3d as o3d
from remimi.sensors.realsense import RealsenseD435i
from remimi.sensors.webcamera import SimpleWebcamera
from remimi.visualizers.point_cloud import SimplePointCloudVisualizer

def get_inverse_map(depth, bits=2):
    depth_min = depth.min()
    depth_max = depth.max()

    # max_val = (2 ** (8 * bits)) - 1

    depth_max = 3000

    # if depth_max - depth_min > np.finfo("float").eps:
    out = depth_max - depth
    # out = max_val - depth
    
    return out

def run(model_path, model_type="dpt_hybrid", optimize=True, use_realsense=False, webcam_id=4, video_file=None):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """

    if use_realsense:
        sensor = RealsenseD435i(resolution=(640, 480))
        intrinsic = sensor.get_open3d_intrinsic()
    else:
        if video_file is not None:
            sensor = SimpleWebcamera(video_file)
        else:
            sensor = SimpleWebcamera(webcam_id)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )

    cam = DPTPaseudoDepthCamera(model_path, sensor, model_type, optimize)
    vis = SimplePointCloudVisualizer()

    print("start processing")
    while True:
        color, depth = cam.get_next_frame()

        # color, depth = sensor.get_color_and_depth()
        
        depth_image = o3d.geometry.Image(depth)
        color_image = o3d.geometry.Image(color)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,                
            depth_scale=1000,
            convert_rgb_to_intensity=False)
        temp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic)

        vis.update_by_pcd(temp)
        # cv2.imshow("Depth", util.io.get_depth(depth, bits=2))
        cv2.imshow("Depth", depth)
        cv2.imshow("color", color)
        key = cv2.waitKey(1)
        if key  == ord('a'):
            vis.stop_update()

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input-file", help="video file to show point cloud."
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument("--show-attention", dest="vis", action="store_true")

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.add_argument("--use-realsense", action="store_true")
    parser.add_argument("--webcam-id", type=int, default=0)
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.model_weights,
        args.model_type,
        args.optimize,
        args.use_realsense,
        args.webcam_id,
        args.input_file
    )
