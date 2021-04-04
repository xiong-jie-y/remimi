from typing import List
from remimi.utils.file import get_model_file
from mmseg.apis import inference_segmentor, init_segmentor
import cv2
import numpy as np
from pkg_resources import resource_filename

class SemanticSegmenter:
    def __init__(self):
        # config_file = resource_filename("remimi", "segmentation/configs/dnlnet/dnl_r101-d8_512x512_80k_ade20k.py")
        # checkpoint_file = get_model_file(
        #     "dln_small.pth", 
        #     "https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_80k_ade20k/dnl_r50-d8_512x512_80k_ade20k_20200826_183354-1cf6e0c1.pth")

        config_file = resource_filename("remimi", "segmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k.py")
        checkpoint_file = get_model_file(
            "deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth",
            "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth"
        )

        # build the model from a config file and a checkpoint file
        model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

        self.model = model


    def convert_to_semantic_image(self, color_image_bgr):
        frame = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        result = inference_segmentor(self.model, frame)
        ret_img = np.zeros(frame.shape)

        # Get Pallete.
        palette = self.model.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.model.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        seg = result[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        return color_seg

    def get_mask(self, color_image_bgr: np.ndarray, class_names: List[str]):
        """Get mask from class_name.

        The class that is in class_names will be black(0).
        Other part will be white(255).
        """
        frame = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        result = inference_segmentor(self.model, frame)
        seg = result[0]
        mask = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
        mask[:,:] = 255
        for class_name in class_names:
            try:
                index = self.model.CLASSES.index(class_name)
            except ValueError:
                print(F"Available classes are {self.model.CLASSES}")
                raise
            mask[seg == index] = 0

        return mask