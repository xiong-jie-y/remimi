"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import open3d as o3d
from remimi.sensors.realsense import RealsenseD435i
from remimi.sensors.webcamera import SimpleWebcamera
from remimi.visualizers.point_cloud import SimplePointCloudVisualizer
from remimi.utils.open3d import create_point_cloud_from_color_and_depth
from remimi.sensors.paseudo_camera import DPTPaseudoDepthCamera

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
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        # )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720, 525.0, 525.0, 639.0, 359.25
        )

    cam = DPTPaseudoDepthCamera(model_path, sensor, model_type, optimize)
    vis = SimplePointCloudVisualizer()

    print("start processing")
    while True:
        color, depth = cam.get_color_and_depth()

        # To see realsense input.
        # color, depth = sensor.get_color_and_depth()
    
        pcd = create_point_cloud_from_color_and_depth(color, depth, intrinsic)

        vis.update_by_pcd(pcd)
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


    # compute depth maps
    run(
        args.model_weights,
        args.model_type,
        args.optimize,
        args.use_realsense,
        args.webcam_id,
        args.input_file
    )
