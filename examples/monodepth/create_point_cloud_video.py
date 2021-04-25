import os
from os.path import basename, join
from remimi.sensors.file import MultipleImageStream
from remimi.edit.hifill.hifill import MaskEliminator
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

def make_stereopair(left, right):
    width, height = left.size
    leftMap = left.load()
    rightMap = right.load()
    pair = Image.new('RGB', (width * 2, height))
    pairMap = pair.load()
    for y in range(0, height):
        for x in range(0, width):
            pairMap[x, y] = leftMap[x, y]
            pairMap[x + width, y] = rightMap[x, y]
    # if color == 'mono':
    #     pair = pair.convert('L')
    return pair


from PIL import Image

def create_mask(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("grayscale left", gray)
    mask[gray != 255] = 255

   #@  import IPython; IPython.embed()

    return mask

from remimi.utils.logging import PerfLogger

def create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, name, parallax):
    logger = PerfLogger(print=False)

    with logger.time_measure("camera"):
        extrinsic = \
            np.array([[ 1.        ,  0.        ,  0.        ,  parallax],
                    [-0.        , -1.        , -0.        ,  0],
                    [-0.        , -0.        , -1.        ,  -0.0001],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])

        pcam = o3d.camera.PinholeCameraParameters()
        pcam.intrinsic = intrinsic
        pcam.extrinsic = extrinsic
        vis.vis.get_view_control().convert_from_pinhole_camera_parameters(pcam)
        vis.vis.get_view_control().set_constant_z_far(10000000)
        vis.vis.get_view_control().set_constant_z_near(-10000000)
    # vis.update_by_pcd(pcd)
    # vis.vis.poll_events()
    # vis.vis.update_renderer()

    # import IPython; IPython.embed()

    with logger.time_measure("capture_image"):
        left_image = (np.array(vis.vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)

    if inpaint:
        # import IPython; IPython.embed()
        left_image_mask = create_mask(left_image).astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)
        # left_image_mask = cv2.erode(left_image_mask,kernel,iterations = 1)
        # import IPython; IPython.embed()
        cv2.imshow("left inpaint mask", left_image_mask)
        mask = cv2.cvtColor(left_image_mask, cv2.COLOR_GRAY2BGR)
        left_inpainted_image = eliminator.eliminate_by_mask(cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR), mask)
        left_image = cv2.cvtColor(left_inpainted_image, cv2.COLOR_BGR2RGB)
    # left_image_r = cv2.ximgproc.weightedMedianFilter(left_image, left_image[:, :, 0], 3, sigma=5, weightType=cv2.ximgproc.WMF_IV1)
    # left_image_g = cv2.ximgproc.weightedMedianFilter(left_image, left_image[:, :, 1], 3, sigma=5, weightType=cv2.ximgproc.WMF_IV1)
    # left_image_b = cv2.ximgproc.weightedMedianFilter(left_image, left_image[:, :, 2], 3, sigma=5, weightType=cv2.ximgproc.WMF_IV1)
    # left_image = np.stack((left_image_r, left_image_g, left_image_b), axis=2)
    # left_image = cv2.medianBlur(left_image,5)

    # cv2.imwrite(join(cache_folder, "{}_{}.png".format(suffix, name)), left_image)
    # cv2.imshow(name, left_image)

    return left_image


# def gallery(array, ncols=3):
#     nindex, height, width, intensity = array.shape
#     nrows = nindex//ncols
#     assert nindex == nrows*ncols
#     # want result.shape = (height*nrows, width*ncols, intensity)
#     result = (array.reshape(nrows, ncols, height, width, intensity)
#               .swapaxes(1,2)
#               .reshape(height*nrows, width*ncols, intensity))
#     return result

def gallery(images, ncols=3):
    # import IPython; IPython.embed()
    nrows = len(images) // ncols
    row_images = []
    for i in range(nrows):
        row_image = np.concatenate(images[i * ncols: (i + 1) * ncols], axis=1)
        row_images.append(row_image)
    
    row_images.reverse()
    # cv2.imshow("kjfkdlsjafklda", np.concatenate(row_images, axis=1))
    return np.concatenate(row_images, axis=0)

@click.command()
@click.option("--video-file")
@click.option("--image-file")
@click.option("--video-url")
@click.option("--cache-root")
@click.option("--model-name", default="ken3d")
@click.option("--save-point-cloud", is_flag=True)
@click.option("--create-anaglyph", is_flag=True)
@click.option("--create-stereo-pair", is_flag=True)
@click.option("--create-looking-glass", is_flag=True)
@click.option("--inpaint", is_flag=True)
@click.option("--debug", is_flag=True)
def run(video_file, image_file, video_url, cache_root, model_name, save_point_cloud, create_anaglyph, create_stereo_pair, debug, inpaint, create_looking_glass):
    if video_url is not None:
        video_file = ensure_video(video_url, cache_root)

    if video_file is not None:
        cache_folder = join(cache_root, basename(video_file))
    elif image_file is not None:
        cache_folder = join(cache_root, "images")
    os.makedirs(cache_folder, exist_ok=True)

    if video_file is not None:
        sensor = SimpleWebcamera(video_file)
        # width, height = 1280, 720
        width, height = 640, 480
    elif image_file is not None:
        sensor = MultipleImageStream([image_file])
        height, width, _ = cv2.imread(image_file).shape
    else:
        sensor = SimpleWebcamera(webcam_id)
    
    base_center_x = 319.5 / 640
    base_center_y = 239.5 / 480

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        # o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        width, height, 1000.0, 1000.0, width / 2 - 0.5, height / 2 - 0.5
        # width, height, 4000.0, 4000.0, width / 2 - 0.5, height / 2 - 0.5
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

    eliminator = MaskEliminator()

    cam = DPTPaseudoDepthCamera(
        sensor, model_name, output_type=ImageType.RGB,boundary_depth_removal=False, debug=debug)
    vis = SimplePointCloudVisualizer((width, height),
        show_axis=False, original_coordinate=False
    )
    vis2 = SimplePointCloudVisualizer((width, height),
        show_axis=False, original_coordinate=False
    )

    #time_logger = PerfLogger(print=True)

    vis.vis.get_render_option().background_color = np.array([0,0,0])
    # skip.
    # for i in range(5150):
    #     sensor.get_color()

    time_logger = PerfLogger(print=False)
    
    vis.vis.get_render_option().point_size = 1.5

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
            # last adjustment
            # For Dance
            baseline = 0.000015
            # baseline = 0.001
            # baseline = 0.000008
            # baseline = 0.00010

            vis.update_by_pcd(pcd)
            left_image = create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, "left", baseline)
            right_image = create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, "right", -baseline)

            ana_image = create_anaglyph_func(Image.fromarray(left_image), Image.fromarray(right_image))
            # import IPython; IPython.embed()
            ana_image_bgr = cv2.cvtColor(np.asarray(ana_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("anaglyph", ana_image_bgr)

            cv2.imwrite(join(cache_folder, "{}_anaglyph.jpg".format(suffix)), ana_image_bgr)

            stereo_image = make_stereopair(Image.fromarray(right_image), Image.fromarray(left_image))
            # import IPython; IPython.embed()
            stereo_image_bgr = cv2.cvtColor(np.asarray(stereo_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("stereo pair", stereo_image_bgr)

            cv2.imwrite(join(cache_folder, "{}_stereo.jpg".format(suffix)), stereo_image_bgr)

        if create_looking_glass:
            # last adjustment
            # For Dance
            # baseline = 0.000015
            # For looking glass
            # A bit thin.
            baseline = 0.000007
            # Closeer camera.
            # baseline = 0.000010
            # baseline = 0.0000017
            # Maybe better?
            # baseline = 0.000013
            # baseline = 0.001
            # baseline = 0.000008
            # baseline = 0.00010

            with time_logger.time_measure("update PCD"):
                vis.update_by_pcd(pcd)        

            with time_logger.time_measure("create_parallax"):
                left_images = [
                    create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, f"num_left_{i}", baseline * i)
                    for i in range(22, 0, -1)
                ]
                right_images = [
                    create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, f"num_right_{i}", -baseline * i)
                    for i in range(0, 22)
                ]

            with time_logger.time_measure("create_gallery"):
                lkimage = gallery(left_images + [
                    create_parallaxed_image(pcd, vis, intrinsic, inpaint, eliminator, cache_folder, suffix, f"num_center", 0)
                ] + right_images, ncols=5)

            cv2.imshow("lkimage", lkimage)

            lkimage_res = cv2.resize(lkimage, (4096, 4096))
    
            cv2.imwrite(join(cache_folder, "{}_{}.png".format(suffix, "lkimage")), cv2.cvtColor(np.asarray(lkimage_res), cv2.COLOR_RGB2BGR))

        # To see realsense input.
        # color, depth = sensor.get_color_and_depth()

        print(f"{frame_no} processing.")

        frame_no += 1

        # vis.update_by_pcd(pcd)

        cv2.imshow("Depth", depth)
        cv2.imshow("color", color)
        key = cv2.waitKey(1)
        # 
        # vis.update_by_pcd(pcd)
        # while True:
        #     vis.vis.poll_events()
        #     vis.vis.update_renderer()
        #     cv2.imshow("Depth", depth)
        #     cv2.imshow("color", color)
        #     key = cv2.waitKey(1)
        if key  == ord('a'):
            vis.stop_update()


if __name__ == '__main__':
    run()