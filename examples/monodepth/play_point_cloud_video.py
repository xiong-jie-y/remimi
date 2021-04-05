import os
from os.path import basename, join
import numpy as np
import cv2

import open3d as o3d
from remimi.sensors.realsense import RealsenseD435i
from remimi.sensors.webcamera import SimpleWebcamera
from remimi.visualizers.point_cloud import SimplePointCloudVisualizer
from remimi.utils.open3d import create_point_cloud_from_color_and_depth
from remimi.sensors.paseudo_camera import DPTPaseudoDepthCamera, ImageType
import click

class ScaleFixer:
    def __init__(self):
        # self.farest_depths = []
        # self.closest_depths = []
        self.farest_ys = []
        self.closest_ys = []
        self.farest_xs = []
        self.closest_xs = []

    def fix_scale(self, pcd):
        points = np.array(pcd.points)
        # farest_depth = np.max(points[:, 2])
        farest_y = np.max(points[:, 1])
        farest_x = np.max(points[:, 0])
        # closest_depth = np.min(points[:, 2])
        closest_y = np.min(points[:, 1])
        closest_x = np.min(points[:, 0])
        if len(self.farest_ys) < 20:
            # self.farest_depths.append(farest_depth)
            # self.closest_depths.append(closest_depth)
            self.farest_ys.append(farest_y)
            self.closest_ys.append(closest_y)
            self.farest_xs.append(farest_x)
            self.closest_xs.append(closest_x)
        else:
            # mean_farest_depth = np.mean(self.farest_depths)
            # mean_closest_depth = np.mean(self.closest_depths)

            # depth_scale_ratio = (mean_farest_depth - mean_closest_depth) / (farest_depth - closest_depth)
            y_scale_ratio = (np.mean(self.farest_ys) - np.mean(self.closest_ys)) / (farest_y - closest_y)
            x_scale_ratio = (np.mean(self.farest_xs) - np.mean(self.closest_xs)) / (farest_x - closest_x)

            adj_scale = np.mean([x_scale_ratio, y_scale_ratio])

            # print("y", y_scale_ratio)
            # print("depth", depth_scale_ratio)

            # points[:, 2] = points[:, 2] + (mean_closest_depth - closest_depth)
            points = adj_scale * points
            pcd.points = o3d.utility.Vector3dVector(points)

class DepthScaleFixer:
    def __init__(self):
        self.farest_depths = []
        self.closest_depths = []
    
    def fix_scale(self, depth):
        farest_depth = np.max(depth)
        closest_depth = np.min(depth)
        if len(self.farest_depths) < 20:
            self.farest_depths.append(farest_depth)
            self.closest_depths.append(closest_depth)
        else:
            mean_closest_depth = np.mean(self.closest_depths)
            mean_farest_depth = np.mean(self.farest_depths)

            depth = depth - (closest_depth - mean_closest_depth)
            depth = (mean_farest_depth - mean_closest_depth) / (farest_depth - closest_depth) * depth

        return depth

import youtube_dl

@click.command()
@click.option("--video-file")
@click.option("--video-url")
@click.option("--cache-root")
@click.option("--model-name", default="ken3d")
def run(video_file, video_url, cache_root, model_name):
    if video_url is not None:
        ydl_opts = {
            "outtmpl": join(cache_root, "videos", "%(id)s.%(ext)s")
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            a = ydl.extract_info(video_url)
            original_video_file = join(cache_root, "videos", f"{a['id']}.{a['ext']}")
            video_file = join(cache_root, "videos", f"{a['id']}_res.mp4")
            
            if not os.path.exists(original_video_file):
                original_video_file = join(cache_root, "videos", f"{a['id']}.mkv")

            import subprocess
            subprocess.run(f"ffmpeg -y -i {original_video_file} -s 640x480  {video_file}", shell=True, check=True)

    cache_folder = join(cache_root, basename(video_file))
    os.makedirs(cache_folder, exist_ok=True)

    fps = None
    if video_file is not None:
        sensor = SimpleWebcamera(video_file)
        fps = sensor.get_fps()
        print(f"fps is {fps}hz.")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        # o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        640, 480, 1000.0, 1000.0, 319.5, 239.5
    )
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     # o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    #     640, 360, 1000.0, 1000.0, 319.5, 179.25
    # )
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     # o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    #     1920, 1080, 1500.0, 1500.0, 958.5, 538.875
    # )
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     # 1280, 720, 525.0, 525.0, 639.0, 359.25
    #     1280, 720, 1500.0, 1500.0, 639.0, 359.25
    # )

    cam = DPTPaseudoDepthCamera(
        sensor, model_name, output_type=ImageType.RGB,boundary_depth_removal=False)
    vis = SimplePointCloudVisualizer(
        show_axis=False, original_coordinate=False
    )
    # vis = InpaintedPointCloudVisualizer()

    vis.vis.get_render_option().point_size = 1

    from pydub import AudioSegment
    from pydub.playback import play
    import threading

    song = song = AudioSegment.from_file(video_file, "mp4")
    def play_ex():
        play(song)

    import time
    x = 0
    frame_no = 0
    print("start processing")
    last_time = time.time() - 100
    t = threading.Thread(target=play_ex)
    t.start()
    scale_fixer = ScaleFixer()
    depth_fixer = DepthScaleFixer()
    while True:
        if fps is not None:
            elapsed = (time.time() - last_time)
            if elapsed < (1./fps):
                continue
            last_time = time.time()

        suffix = str(frame_no).zfill(6)

        color = cv2.imread(join(cache_folder, "{}_color.jpg".format(str(frame_no).zfill(6))))
        file2 = join(cache_folder, "{}_depth.png".format(suffix))

        if os.path.exists(file2):
            depth = cv2.imread(file2, -1)
        else:
            depth = np.load(join(cache_folder, "{}_depth.npy".format(suffix)))

        # import IPython; IPython.embed()
        # depth = depth_fixer.fix_scale(depth)

        pcd = create_point_cloud_from_color_and_depth(color, depth, intrinsic)
        # ind = np.load(join(cache_folder, "{}_outlier.npy".format(suffix)))
        # pcd = o3d.io.read_point_cloud(join(cache_folder, "{}.ply".format(str(frame_no).zfill(6))))

        # pcd = pcd.voxel_down_sample(voxel_size=0.0008)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                         std_ratio=1.0)
        # inlier_cloud = pcd.select_by_index(ind)
        # print(len(np.array(pcd.points)))
        # o3d.io.write_point_cloud(join(cache_folder, "{}.ply".format(str(frame_no).zfill(6))), inlier_cloud)
        # pcd = inlier_cloud
 
        scale_fixer.fix_scale(pcd)

        vis.update_by_pcd(pcd)

        # extrinsic = \
        #     np.array([[ 1.        ,  0.        ,  0.        ,  x],
        #             [-0.        , -1.        , -0.        ,  0],
        #             [-0.        , -0.        , -1.        ,  0],
        #             [ 0.        ,  0.        ,  0.        ,  1.        ]])

        # pcam = o3d.camera.PinholeCameraParameters()
        # pcam.intrinsic = intrinsic
        # pcam.extrinsic = extrinsic
        # vis.vis.get_view_control().convert_from_pinhole_camera_parameters(pcam)

        # x -= 0.01

        # import IPython; IPython.embed()
        frame_no += 1

        # if not use_cache:
        cv2.imshow("Depth", depth)
        cv2.imshow("color", color)
        key = cv2.waitKey(1)
        if key  == ord('a'):
            vis.stop_update()


if __name__ == '__main__':
    run()