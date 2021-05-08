from matplotlib import pyplot as plt
from remimi.utils.image import show_image_ui
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
    depth = cv2.normalize(image, None, 0, 2 ** 16, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth

def zero_crossing_v2(LoG):
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
    return zeroCross

def statistical_zero_crossing(laplacian_image):
    """zero crossing edge detection that takes 90% of bigger sloop edge."""
    minLoG = cv2.morphologyEx(laplacian_image, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(laplacian_image, cv2.MORPH_DILATE, np.ones((3,3)))
    sloop = maxLoG - minLoG
    zeroCross = np.logical_and(
        # This is the approximate solution to set t he minimum sloop.
        (sloop > np.percentile(sloop, 95)),
        np.logical_or(
            np.logical_and(minLoG < 0,  laplacian_image > 0),
            np.logical_and(maxLoG > 0, laplacian_image < 0))
    )
    gray = np.zeros(zeroCross.shape, dtype=np.uint8)
    gray[zeroCross] = 255
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2,2)))

def remove_outlier(depth_image):
    median = np.median(depth_image)
    depth_image[
        np.abs(depth_image - median) > np.std(depth_image) * 3
    ] = 0
    return depth_image

def clip_depth_to_foreground(depth_image, foreground_rate = 20):
    depth_median = np.percentile(depth_image, foreground_rate)
    depth_mean_diff = np.median(np.abs(depth_image - depth_median))
    depth_min = depth_median - depth_mean_diff
    depth_max = depth_median + depth_mean_diff
    depth_image = depth_image.clip(depth_min, depth_max)

    return depth_image

def detect_edge(depth_image, zero_crossing_method):
    depth_image = clip_depth_to_foreground(depth_image)
    depth_image = cv2.GaussianBlur(depth_image,(3,3),8)
    edge_image = cv2.Laplacian(depth_image, cv2.CV_64F)

    return zero_crossing_method(edge_image)

def create_foreground_mask_v2(depth_image, debug=False):
    depth_image = clip_depth_to_foreground(depth_image)
    # depth_image = np.clip(depth_image, 0, np.percentile(depth_image, 50))
    depth_image_u16 = cv2.normalize(depth_image, None, 0, 2 ** 16 - 1, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    _, mask = cv2.threshold(
        depth_image_u16,
        np.min(depth_image_u16), 
        np.max(depth_image_u16), 
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    mask_u8 = cv2.bitwise_not(mask).astype(np.uint8)
    mean_masked = np.mean(depth_image_u16[mask_u8 == 255])
    close_to_mean = depth_image_u16 - mean_masked < np.std(depth_image_u16[mask_u8 == 255]) / 2

    foreground_mask = np.zeros(depth_image.shape, dtype=np.uint8)
    foreground_mask[close_to_mean] = 255

    # type cast is necessary because depth is u16.
    return foreground_mask

def get_foreground_background_edges(depth_image, debug=False):
    edge_image = detect_edge(depth_image, statistical_zero_crossing)

    depth_image = clip_depth_to_foreground(depth_image)
    depth_image_u16 = cv2.normalize(depth_image, None, 0, 2 ** 16, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    _, labels = cv2.connectedComponents(edge_image)

    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        one_edge_image = np.zeros(edge_image.shape, dtype=np.uint8)
        one_edge_image[labels == label_id] = 255

        dilated_edge_for_one_image = cv2.morphologyEx(
            one_edge_image, cv2.MORPH_DILATE, np.ones((3,3)), iterations=30)
        if debug:
            show_image_ui(dilated_edge_for_one_image, cmap=plt.cm.gray)

        depth_in_the_region = depth_image_u16[dilated_edge_for_one_image == 255]
        fore_back_threash, _ = cv2.threshold(
            depth_in_the_region,
            np.min(depth_in_the_region), 
            np.max(depth_in_the_region), 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        foreground_slice = np.bitwise_and(
            dilated_edge_for_one_image == 255, depth_image_u16 < fore_back_threash)
        background_slice = np.bitwise_and(
            dilated_edge_for_one_image == 255, depth_image_u16 >= fore_back_threash)
        yield foreground_slice, background_slice, one_edge_image

def create_foreground_mask(depth_image, debug=False):
    foreground_mask = np.zeros(depth_image.shape, dtype=np.uint8)

    for foreground_slice, _, _ in get_foreground_background_edges(depth_image):
        foreground_image = np.zeros(depth_image.shape, dtype=np.uint8)
        foreground_image[foreground_slice] = 255
        if debug:
            show_image_ui(foreground_image, cmap=plt.cm.gray)

        foreground_mask[foreground_slice] = 255

    return foreground_mask

class DPTDepthImageContainer:
    def __init__(self, original_size, inverse_depth_image: torch.Tensor):
        """Create container with raw output of DPT.
        
        The definition of inverse depth is correspond to the definiton in stereo depth estimation.
        The value is supposed to the direct output of DPT.

        Arguments:
            - original_size: (H, W)
            - inverse_depth_image: (1, H, W) inverse depth and this is susupposed to on some cuda core.
        """
        self.inverse_depth_image = inverse_depth_image
        self.original_size = original_size

    def get_depth_image(self):
        """Get depth image from a inverse depth images.

        Returns: 
            Depth image. The infinity is cliped to max value in the depth.
        """
        inverse_depth_image = self.inverse_depth_image
        inverse_depth_image = inverse_depth_image.unsqueeze(0).double()

        scale_ratio = 1.0

        inverse_depth_image = torch.nn.functional.interpolate(
            inverse_depth_image, size=self.original_size, mode='bilinear', align_corners=False
        )[0, 0, :, :].cpu().numpy() * scale_ratio

        depth_image = 1.0 / inverse_depth_image

        # To replace infinity.
        finite_depth_image = depth_image.clip(np.nanmin(depth_image), np.nanmax(depth_image[depth_image != np.inf]))

        return finite_depth_image

    def get_inverse_depth_image(self):
        resized_disparity = torch.nn.functional.interpolate(
            self.inverse_depth_image.unsqueeze(1),
            size=self.original_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return resized_disparity