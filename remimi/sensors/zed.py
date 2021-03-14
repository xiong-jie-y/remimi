import os
from os.path import join
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
import open3d as o3d
import click

class ZedCamera:
    def __init__(self):
        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        init_params.camera_resolution = sl.RESOLUTION.HD720

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        self.runtime_parameters = runtime_parameters

        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.pose = sl.Pose()

        tracking_parameters = sl.PositionalTrackingParameters()
        err = zed.enable_positional_tracking(tracking_parameters)
        if (err != sl.ERROR_CODE.SUCCESS):
            exit(-1)

        mapping_parameters = sl.SpatialMappingParameters()
        err = zed.enable_spatial_mapping(mapping_parameters)
        if (err != sl.ERROR_CODE.SUCCESS):
            exit(-1)

        self.zed = zed

    def _get_open3d_depth(self, depth_image):
        depth_image = np.nan_to_num(depth_image, 0)
        depth_image[depth_image == -np.inf] = 0
        depth_image[depth_image == np.inf] = 0
        depth_image *= 1000
        depth_image = depth_image.astype(np.uint16)
        return depth_image

    def get_rgb_and_depth(self):
        # A new image is available if grab() returns SUCCESS
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

            # self.width = self.image.get_width()
            # self.height = self.image.get_height()

            return self.image.get_data(),  self._get_open3d_depth(np.copy(self.depth.get_data()))
        else:
            return None

    def get_data(self):
        # A new image is available if grab() returns SUCCESS
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

            tracking_state = self.zed.get_position(self.pose)
            pose = None
            if tracking_state.name == "SEARCHING":
                pose = None
            else:
                print(tracking_state)
                pose = np.zeros((4,4))
                pose[:3,:3] = self.pose.get_rotation_matrix().r
                pose[:3, 3] = self.pose.get_translation().get()
                pose[3, 3] = 1

            # import IPython; IPython.embed()
            return np.copy(self.image.get_data()),  self._get_open3d_depth(np.copy(self.depth.get_data())), pose
        else:
            return None



    def get_intrinsic(self):
        camera_info = self.zed.get_camera_information()
        cam_params = camera_info.calibration_parameters
        K = np.array([[cam_params.left_cam.fx,                      0, cam_params.left_cam.cx],
                    [                     0, cam_params.left_cam.fy, cam_params.left_cam.cy],
                    [                     0,                      0,                      1]])
        return K
    
    def get_open3d_intrinsic(self):
        camera_info = self.zed.get_camera_information()
        cam_params = camera_info.calibration_parameters
        left_camera = cam_params.left_cam
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720, left_camera.fx,
                                            left_camera.fy, left_camera.cx,
                                            left_camera.cy)
        print(intrinsic.intrinsic_matrix)
        return intrinsic