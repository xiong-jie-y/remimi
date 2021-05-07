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


class DPTDepthImageContainer:
    def __init__(self, original_size, inverse_depth_image: torch.Tensor):
        """Create container with raw output of DPT.
        
        The definition of inverse depth is correspond to the definiton in stereo depth estimation.
        The value is supposed to the direct output of DPT.

        Arguments:
            - inverse_depth_image: (1, H, W) inverse depth and this is susupposed to on some cuda core.
        """
        self.inverse_depth_image = inverse_depth_image

    def get_depth_image(self):
        """Get depth image from a inverse depth images.

        Returns: 
            Depth image. The infinity is cliped to max value in the depth.
        """
        inverse_depth_image = self.inverse_depth_image
        inverse_depth_image = inverse_depth_image.unsqueeze(0).double()

        scale_ratio = 1.0

        inverse_depth_image = torch.nn.functional.interpolate(
            inverse_depth_image, size=(480, 640), mode='bilinear', align_corners=False
        )[0, 0, :, :].cpu().numpy() * scale_ratio

        depth_image = 1.0 / inverse_depth_image

        # To replace infinity.
        finite_depth_image = depth_image.clip(np.nanmin(depth_image), np.nanmax(depth_image[depth_image != np.inf]))

        return finite_depth_image

    def get_inverse_depth_image(self):
        return self.inverse_depth_image