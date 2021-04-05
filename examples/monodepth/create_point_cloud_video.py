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

    if video_file is not None:
        sensor = SimpleWebcamera(video_file)
    else:
        sensor = SimpleWebcamera(webcam_id)
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

    vis.vis.get_render_option().point_size = 1

    import time
    x = 0
    frame_no = 0
    print("start processing")
    last_time = time.time() - 100
    while True:
        suffix = str(frame_no).zfill(6)

        color, depth = cam.get_color_and_depth()

        # depth = depth.astype(np.uint16)
        cv2.imwrite(join(cache_folder, "{}_color.jpg".format(suffix)), color)
        # cv2.imwrite(join(cache_folder, "{}_depth.png".format(str(frame_no).zfill(6))), depth)
        np.save(join(cache_folder, "{}_depth".format(suffix)), depth)

        pcd = create_point_cloud_from_color_and_depth(color, depth, intrinsic)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                         std_ratio=1.0)
        # np.save(join(cache_folder, "{}_outlier".format(suffix)), ind)

        # To see realsense input.
        # color, depth = sensor.get_color_and_depth()

        frame_no += 1

        vis.update_by_pcd(pcd)

        cv2.imshow("Depth", depth)
        cv2.imshow("color", color)
        key = cv2.waitKey(1)
        if key  == ord('a'):
            vis.stop_update()


if __name__ == '__main__':
    run()