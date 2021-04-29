from configs.quilit_video_generator import DEFAULT_OPTIONS
import glob
import os
from os.path import basename, join
from remimi.generators.parallax_video_generator import ParallaxVideoGenerator
from remimi.sensors.edit_stream import CustomizableHumanEliminatedStream, HumanEliminatedStream
from remimi.monodepth.ken3d.depthestim import Ken3DDepthEstimator
from remimi.detection.instance_segmentation import InstanceSegmenter
from remimi.utils.depth import colorize2
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

@click.command()
@click.option("--video-file")
@click.option("--background-video-file")
@click.option("--image-file")
@click.option("--mask-dir")
@click.option("--video-url")
@click.option("--cache-root")
@click.option("--model-name", default="ken3d")
@click.option("--option-name", default="looking_glass_8_9_inpaint_option")
@click.option("--save-point-cloud", is_flag=True)
@click.option("--create-anaglyph", is_flag=True)
@click.option("--create-stereo-pair", is_flag=True)
@click.option("--create-looking-glass", is_flag=True)
@click.option("--debug", is_flag=True)
def run(video_file, background_video_file, mask_dir, image_file, video_url, cache_root, model_name, save_point_cloud, create_anaglyph, create_stereo_pair, debug, create_looking_glass, option_name):
    if video_url is not None:
        video_file = ensure_video(video_url, cache_root)

    if video_file is not None:
        cache_folder = join(cache_root, basename(video_file))
    elif image_file is not None:
        cache_folder = join(cache_root, "images")
    os.makedirs(cache_folder, exist_ok=True)

    generator = ParallaxVideoGenerator(DEFAULT_OPTIONS[option_name])
    generator.generate(cache_folder, video_file, background_video_file, mask_dir, image_file, video_url, cache_root, model_name, save_point_cloud, create_anaglyph, create_stereo_pair, debug, create_looking_glass)


if __name__ == '__main__':
    run()