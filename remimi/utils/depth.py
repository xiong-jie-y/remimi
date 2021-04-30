from typing import Optional, Tuple
import numpy as np
import cv2
import torch

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

def colorize2(image):
    depth = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth

def get_depth(disparity_image):
    """

    Arguments:
        - disparity_image: (H, W) disparity.
    """
    baseline = 40
    focal_length = max(disparity_image.shape[1], disparity_image.shape[2]) / 2.0

    disparity_image = disparity_image.unsqueeze(0)

    # import IPython; IPython.embed()
    disparity_image = torch.nn.functional.interpolate(
        disparity_image, size=(480, 640), mode='bilinear'
    ) * (max(disparity_image.shape[0], disparity_image.shape[1]) / 256.0)

    # to avoid zero division.
    # disparity_image[disparity_image == 0] = 0.0000001
    depth_image = ((focal_length * baseline) / disparity_image + 0.0000001)[0, 0, :, :].cpu().numpy()
    # 255 is randomly chosen not related to 8bit.
    depth_image[depth_image > 40000] = 40000

    return depth_image.astype(np.float32)

# class DepthImage:
    