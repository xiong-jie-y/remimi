import os
from os.path import basename, join
from remimi.utils.file import ensure_video
import numpy as np
import cv2

import open3d as o3d
from remimi.sensors.realsense import RealsenseD435i
from remimi.sensors.webcamera import SimpleWebcamera
from remimi.visualizers.point_cloud import SimplePointCloudVisualizer, StereoImageVisualizer
from remimi.utils.open3d import create_point_cloud_from_color_and_depth
from remimi.sensors.paseudo_camera import DPTPaseudoDepthCamera, ImageType
from remimi.sensors import StreamFinished
import click

import youtube_dl

def create_anaglyph_func(left, right):
    m = [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114 ] ]
   #  m = [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114 ] ]
    # m = [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ]
    # m = [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ]
    m = [ [ 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ]
    width, height = left.size
    leftMap = left.load()
    rightMap = right.load()

    for y in range(0, height):
        for x in range(0, width):
            r1, g1, b1 = leftMap[x, y]
            r2, g2, b2 = rightMap[x, y]
            leftMap[x, y] = (
                int(r1*m[0][0] + g1*m[0][1] + b1*m[0][2] + r2*m[1][0] + g2*m[1][1] + b2*m[1][2]),
                int(r1*m[0][3] + g1*m[0][4] + b1*m[0][5] + r2*m[1][3] + g2*m[1][4] + b2*m[1][5]),
                int(r1*m[0][6] + g1*m[0][7] + b1*m[0][8] + r2*m[1][6] + g2*m[1][7] + b2*m[1][8])
            )

    return left

from PIL import Image

@click.command()
@click.option("--video-file")
@click.option("--video-url")
@click.option("--cache-root")
@click.option("--model-name", default="ken3d")
@click.option("--save-point-cloud", is_flag=True)
@click.option("--create-anaglyph", is_flag=True)
@click.option("--debug", is_flag=True)
def run(video_file, video_url, cache_root, model_name, save_point_cloud, create_anaglyph, debug):
    if video_url is not None:
        video_file = ensure_video(video_url, cache_root)

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
        sensor, model_name, output_type=ImageType.RGB,boundary_depth_removal=False, debug=debug)
    vis = SimplePointCloudVisualizer(
        show_axis=False, original_coordinate=False
    )
    # skip.
    # for i in range(5150):
    #     sensor.get_color()

    
    vis.vis.get_render_option().point_size = 1

    import time
    x = 0
    frame_no = 0
    print("start processing")
    last_time = time.time() - 100
    while True:
        suffix = str(frame_no).zfill(6)

        try:
            color, depth = cam.get_color_and_depth()
            # depth *= 1000
        except StreamFinished:
            print("Finished")
            break

        # depth = depth.astype(np.uint16)
        cv2.imwrite(join(cache_folder, "{}_color.jpg".format(suffix)), color)
        # cv2.imwrite(join(cache_folder, "{}_depth.png".format(str(frame_no).zfill(6))), depth)
        np.save(join(cache_folder, "{}_depth".format(suffix)), depth)


        # color = cv2.imread(join(cache_folder, "{}_color.jpg".format(str(frame_no).zfill(6))))
        # file2 = join(cache_folder, "{}_depth.png".format(suffix))

        # if os.path.exists(file2):
        #     depth = cv2.imread(file2, -1)
        # else:
        #     depth = np.load(join(cache_folder, "{}_depth.npy".format(suffix)))


        pcd = create_point_cloud_from_color_and_depth(color, depth, intrinsic)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                         std_ratio=1.0)
        # np.save(join(cache_folder, "{}_outlier".format(suffix)), ind)

        if save_point_cloud:
            o3d.io.write_point_cloud(join(cache_folder, "{}.ply".format(str(frame_no).zfill(6))), pcd)

        if create_anaglyph:
            baseline = 0.000015
            # baseline = 0.00010
            x = baseline
            vis.update_by_pcd(pcd)
            extrinsic = \
                np.array([[ 1.        ,  0.        ,  0.        ,  x],
                        [-0.        , -1.        , -0.        ,  0],
                        [-0.        , -0.        , -1.        ,  0],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])

            pcam = o3d.camera.PinholeCameraParameters()
            pcam.intrinsic = intrinsic
            pcam.extrinsic = extrinsic
            vis.vis.get_view_control().convert_from_pinhole_camera_parameters(pcam)
            # vis.update_by_pcd(pcd)

            left_image = (np.array(vis.vis.capture_screen_float_buffer(False)) * 255).astype(np.uint8)
            # import IPython; IPython.embed()
            cv2.imshow("left", left_image)

            vis.vis.poll_events()
            vis.vis.update_renderer()
            vis.vis.poll_events()
            vis.vis.update_renderer()
            # vis.update_by_pcd(pcd)
            x = -baseline
            # vis.update_by_pcd(pcd)
            extrinsic = \
                np.array([[ 1.        ,  0.        ,  0.        ,  x],
                        [-0.        , -1.        , -0.        ,  0],
                        [-0.        , -0.        , -1.        ,  0],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])

            pcam = o3d.camera.PinholeCameraParameters()
            pcam.intrinsic = intrinsic
            pcam.extrinsic = extrinsic
            vis.vis.get_view_control().convert_from_pinhole_camera_parameters(pcam)
            # vis.update_by_pcd(pcd)

            right_image = (np.array(vis.vis.capture_screen_float_buffer(False)) * 255).astype(np.uint8)
            # import IPython; IPython.embed()
            cv2.imshow("right", right_image)

            ana_image = create_anaglyph_func(Image.fromarray(right_image), Image.fromarray(left_image))
            # import IPython; IPython.embed()
            ana_image_bgr = cv2.cvtColor(np.asarray(ana_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("anaglyph", ana_image_bgr)

            cv2.imwrite(join(cache_folder, "{}_anaglyph.jpg".format(suffix)), ana_image_bgr)


        # To see realsense input.
        # color, depth = sensor.get_color_and_depth()

        print(f"{frame_no} processing.")

        frame_no += 1

        # vis.update_by_pcd(pcd)

        cv2.imshow("Depth", depth)
        cv2.imshow("color", color)
        key = cv2.waitKey(1)
        if key  == ord('a'):
            vis.stop_update()


if __name__ == '__main__':
    run()