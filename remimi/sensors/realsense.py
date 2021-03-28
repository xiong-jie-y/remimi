import numpy as np
import pyrealsense2 as rs
import open3d as o3d

class RealsenseD435i():
    def __init__(self, resolution=(848, 480)):
        widht, height = resolution
        config = rs.config()
        config.enable_stream(rs.stream.color, widht, height, rs.format.bgr8, 60)
        #
        config.enable_stream(rs.stream.depth, widht, height, rs.format.z16, 60)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        self.run_config=  profile
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        scale=  depth_sensor.get_depth_scale()
        print(f"scale: {scale}")

        self.resolution = resolution

    def get_color(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        return np.array(color_frame.get_data())

    def get_color_and_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        return np.array(color_frame.get_data()), np.array(depth_frame.get_data())

    def get_data(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        acc = frames[2].as_motion_frame().get_motion_data()
        gyro = frames[3].as_motion_frame().get_motion_data()
        timestamp = frames[3].as_motion_frame().get_timestamp()

        return color_frame, depth_frame, acc, gyro, timestamp

    def get_intrinsic(self):
        video_profile = self.run_config.get_stream(rs.stream.color)
        intri = video_profile.as_video_stream_profile().get_intrinsics()
        return intri

    def get_open3d_intrinsic(self):
        video_profile = self.run_config.get_stream(rs.stream.color)
        intrinsics = video_profile.as_video_stream_profile().intrinsics
        out = o3d.camera.PinholeCameraIntrinsic(self.resolution[0], self.resolution[1], intrinsics.fx,
                                                intrinsics.fy, intrinsics.ppx,
                                                intrinsics.ppy)
        return out