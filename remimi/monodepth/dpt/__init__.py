import os
import glob
from remimi.utils.depth import DPTDepthImageContainer
from remimi.segmentation.rgb_segmentation import SemanticSegmenter
import torch
import cv2
import argparse
import numpy as np
import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from util.misc import visualize_attention

def get_inverse_map(depth, bits=2):
    depth_min = depth.min()
    depth_max = depth.max()

    # max_val = (2 ** (8 * bits)) - 1

    depth_max = 3000

    # if depth_max - depth_min > np.finfo("float").eps:
    out = depth_max - depth
    # out = max_val - depth
    
    return out

import enum
class ImageType(enum.Enum):
    RGB = "rgb"
    BGR = "bgr"

from remimi.utils.file import get_model_file

class DPTDepthEstimator:
    def __init__(self, debug=False):
        optimize = True
        model_type="dpt_hybrid"
        self.model_type = model_type
        default_models = {
            "midas_v21": "weights/midas_v21-f6b98070.pt",
            "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        }
        model_path = get_model_file(
            "dpt_hybrid-midas-501f0c75.pt",
            "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
        )

        print("initialize")

        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % device)

        net_w = net_h = 384

        # load network
        if model_type == "dpt_large":  # DPT-Large
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=debug,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=debug,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "midas_v21":  # Convolutional model
            model = MidasNet_large(model_path, non_negative=True)
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|midas_v21]"

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize == True and device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(device)

        self.model = model
        self.transform = transform
        self.device = device
        self.optimize = optimize

        # self.cap = cv2.VideoCapture(camera_id)
        self.debug = debug

    def estimate_depth_raw(self, color):
        img_input = self.transform({"image": color})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            
            return prediction
            # if self.debug:
            #     visualize_attention(sample, self.model, prediction, self.model_type)

        return prediction

    def estimate_depth(self, color):
        prediction = self.estimate_depth_raw(color)

        prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=color.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = get_inverse_map(prediction.astype(np.uint16))
        return depth

    def estimate_and_get_depth_image_container(self, color_image):
        return DPTDepthImageContainer(
            color_image.shape,
            self.estimate_depth_raw(color_image))