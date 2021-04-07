from remimi.utils.depth import colorize
from typing import Optional, Tuple
import pyk4a
import open3d as o3d
from pyk4a import Config, PyK4A
import cv2

import numpy as np

class AzureKinectCamera:
    def __init__(self):
        k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,))
        k4a.start()
        k4a.save_calibration_json('intrinsics.json')

        self.k4a = k4a
        self.previous_capture = None

    def capture(self):
        self.previous_capture = self.k4a.get_capture()

    def get_color_and_depth(self):
        capture = self.previous_capture
        return capture.color, capture.transformed_depth

    def get_rgbd(self, show_raw_input=False):
        capture = self.previous_capture
        if show_raw_input:
            if capture.color is not None     :
                cv2.imshow("Color", capture.color)
            if capture.transformed_depth is not None:
                cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))

        return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(capture.color), o3d.geometry.Image(capture.transformed_depth))

    def get_point_cloud(self, show_raw_input=False):
        rgbd_image = self.get_rgbd(show_raw_input)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                1280, 720, 604.6009521484375, 604.33746337890625, 637.83380126953125, 366.49652099609375
            )
        )

        return pcd