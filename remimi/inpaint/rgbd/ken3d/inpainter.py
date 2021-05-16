import dataclasses
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from remimi.monodepth.ken3d.pointcloud_inpainting import pointcloud_inpainting
from remimi.utils.depth import DPTDepthImageContainer, create_foreground_mask, create_roi_from_foreground_and_background_slice, get_foreground_background_edges
from remimi.utils.image import show_image_ui


import streamlit as st
import enum
import simple_parsing

class OccludingObjectsInpaintMethod(enum.Enum):
    EdgeByEdge = "edge_by_edge"
    OtsuMask = "otsu_mask"

@dataclasses.dataclass
class JointRGBAndDepthOption:
    inpaint_method: OccludingObjectsInpaintMethod = OccludingObjectsInpaintMethod.EdgeByEdge


class JointRGBAndDepthInpainter:
    """Inpainter of the missing region of rgbd image.

    This method is based on the RGBD inpainting method described in 
    '3d ken burns effect from a single image'.
    """
    def __init__(self, options):
        self.options = options

    @classmethod
    def add_arguments(cls, parser: simple_parsing.ArgumentParser):
        parser.add_arguments(JointRGBAndDepthOption, dest="joint_rgbd")
    
    @classmethod
    def to_options(cls, arguments):
        return arguments.joint_rgbd

    def __inpaint_with_inpaint_mask(self, rgb_image, depth_image, disparity_image, inpaint_mask_image):
        rgb_tensor = torch.FloatTensor(rgb_image / 255.).permute(2, 0, 1)[None, ...].contiguous().cuda()
        disparity_tensor = torch.FloatTensor(disparity_image / np.max(disparity_image))[None, None, ...].contiguous().cuda()
        inpaint_mask_tensor = torch.FloatTensor(inpaint_mask_image)[None, None, ...].contiguous().cuda()

        dictionary = pointcloud_inpainting(rgb_tensor, disparity_tensor, inpaint_mask_tensor)

        inpainted_rgb = dictionary['tenImage'].squeeze(0).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        inpainted_disparity = dictionary['tenDisparity'].squeeze(0).squeeze(0).cpu().detach().numpy().copy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        output_rgb_image = cv2.cvtColor(
                inpainted_rgb,
                cv2.COLOR_BGR2RGB
        )
        show_image_ui(rgb_image)
        show_image_ui(disparity_image / np.max(disparity_image))
        show_image_ui(inpaint_mask_image, cmap=plt.cm.gray)
        show_image_ui(output_rgb_image)
        show_image_ui(inpainted_disparity)

        return output_rgb_image, inpainted_disparity * np.max(disparity_image)


    def inpaint_occluding_objects(self, rgb_image, depth_image_container: DPTDepthImageContainer):
        """Inpaint occluded region.

        Arguments:
            - rgb_image: rgb image.
            - depth_image_container: the depth image container
        Returns: 
            Inpainted image.
        """

        depth_image = depth_image_container.get_depth_image()
        disparity_image = depth_image_container.get_inverse_depth_image()

        if self.options.inpaint_method == OccludingObjectsInpaintMethod.EdgeByEdge:
            combined_inpainted_rgb_image = rgb_image.copy()
            combined_inpainted_disparity_image = disparity_image.copy()

            for i, (foreground_slice, background_slice, edge) in enumerate(get_foreground_background_edges(depth_image)):
                box = create_roi_from_foreground_and_background_slice(foreground_slice, background_slice, margin=30)

                context_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                context_rgb_image = context_rgb_image[box]
                context_depth_image = depth_image[box]
                context_disparity_image = disparity_image[box]

                inpaint_mask_image = np.ones(depth_image.shape)
                inpaint_mask_image[foreground_slice] = 0
                inpaint_mask_image = cv2.erode(inpaint_mask_image, kernel=np.ones((3,3)), iterations=5)
                inpaint_mask_image = inpaint_mask_image[box]

                inpainted_rgb_image, inpainted_disparity_image = self.__inpaint_with_inpaint_mask(context_rgb_image, context_depth_image, context_disparity_image, inpaint_mask_image)
                
                combined_inpainted_rgb_image[box][inpaint_mask_image == 0] = inpainted_rgb_image[inpaint_mask_image == 0] * 255
                combined_inpainted_disparity_image[box][inpaint_mask_image == 0] = inpainted_disparity_image[inpaint_mask_image == 0]
        
            show_image_ui(combined_inpainted_rgb_image)
            show_image_ui(combined_inpainted_disparity_image)
        
        elif self.options.inpaint_method == OccludingObjectsInpaintMethod.OtsuMask:
            foreground_mask = create_foreground_mask(depth_image)
            inpaint_mask_image = np.ones(depth_image.shape)
            inpaint_mask_image[foreground_mask == 255] = 0
            self.__inpaint_with_inpaint_mask(rgb_image, depth_image, disparity_image, inpaint_mask_image)