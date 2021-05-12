import cv2
import numpy as np
import torch
from remimi.monodepth.ken3d.pointcloud_inpainting import pointcloud_inpainting
from remimi.utils.depth import DPTDepthImageContainer, create_foreground_mask
from remimi.utils.image import show_image_ui


class JointRGBAndDepthInpainter:
    """Inpainter of the missing region of rgbd image.

    This method is based on the RGBD inpainting method described in 
    '3d ken burns effect from a single image'.
    """
    def __init__(self):
        pass

    def inpaint(self, rgb_image, depth_image_container: DPTDepthImageContainer):
        """Inpaint occluded region.

        Arguments:
            - rgb_image: rgb image.
            - depth_image_container: the depth image container
        Returns: 
            Inpainted image.
        """

        depth_image = depth_image_container.get_depth_image()
        disparity_image = depth_image_container.get_inverse_depth_image()

        foreground_mask = create_foreground_mask(depth_image)

        # input should be RGB.
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        inpaint_mask_image = np.ones(depth_image.shape)
        inpaint_mask_image[foreground_mask == 255] = 0
        rgb_tensor = torch.FloatTensor(rgb_image / 255.).permute(2, 0, 1)[None, ...].contiguous().cuda()
        disparity_tensor = torch.FloatTensor(disparity_image / np.max(disparity_image))[None, None, ...].contiguous().cuda()
        inpaint_mask_tensor = torch.FloatTensor(inpaint_mask_image)[None, None, ...].contiguous().cuda()

        dictionary = pointcloud_inpainting(rgb_tensor, disparity_tensor, inpaint_mask_tensor)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        output_rgb_image = cv2.cvtColor(
                dictionary['tenImage'].squeeze(0).squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
                cv2.COLOR_BGR2RGB
        )
        show_image_ui(rgb_image)
        show_image_ui(disparity_image / np.max(disparity_image))
        show_image_ui(inpaint_mask_image)
        show_image_ui(output_rgb_image)
        show_image_ui(dictionary['tenDisparity'].squeeze(0).squeeze(0).cpu().detach().numpy().copy())
