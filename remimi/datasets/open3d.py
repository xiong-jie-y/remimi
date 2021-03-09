import open3d as o3d

import os
import glob
import numpy as np
import json

class Open3DReconstructionDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.len_frame = len(list(glob.glob(os.path.join(root_dir, "color/*.jpg"))))

    def get_rgb_paths(self):
        open3d_rgb_paths = []
        for i in range(0, self.len_frame):    
            open3d_rgb_paths.append(os.path.join(self.root_dir, "color", '{:06}.jpg'.format(i)))

        return open3d_rgb_paths

    def get_depth_paths(self):
        open3d_depth_paths = []
        for i in range(0, self.len_frame):
            open3d_depth_paths.append(os.path.join(self.root_dir, "depth", '{:06}.png'.format(i)))
        return open3d_depth_paths
    
    def get_trajectory(self):
        lines = open(os.path.join(self.root_dir, "scene/trajectory.log"), 'r').readlines()
        mats = []
        for i in range(0, self.len_frame * 5, 5):
            rows = [
                [float(t) for t in lines[i + 1].split(" ")],
                [float(t) for t in lines[i + 2].split(" ")],
                [float(t) for t in lines[i + 3].split(" ")],
                [float(t) for t in lines[i + 4].split(" ")]
            ]
            mats.append(np.array(rows))
        return mats

    def get_intrinsic(self, type = "raw"):
        if type == "raw":
            return json.load(open(os.path.join(self.root_dir, "camera_intrinsic.json")))
        elif type == "open3d":
            intrinsics = json.load(open(os.path.join(self.root_dir, "camera_intrinsic.json")))
            return o3d.camera.PinholeCameraIntrinsic(
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["intrinsic_matrix"][0],
                intrinsics["intrinsic_matrix"][4],
                intrinsics["intrinsic_matrix"][6],
                intrinsics["intrinsic_matrix"][7],
            )