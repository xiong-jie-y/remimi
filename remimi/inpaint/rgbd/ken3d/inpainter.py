import dataclasses
from remimi.segmentation.u2net_wrapper import U2MaskModel

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from remimi.monodepth.ken3d.pointcloud_inpainting import pointcloud_inpainting
from remimi.utils.depth import DPTDepthImageContainer, create_foreground_mask, create_roi_from_foreground_and_background_slice, create_roi_from_u8_mask, get_foreground_background_edges
from remimi.utils.image import show_image_ui


import streamlit as st
import enum
import simple_parsing

class OccludingObjectsInpaintMethod(enum.Enum):
    EdgeByEdge = "edge_by_edge"
    OtsuMask = "otsu_mask"
    U2NetPretrained = "u2_net_pretrained"

@dataclasses.dataclass
class JointRGBAndDepthOption:
    inpaint_method: OccludingObjectsInpaintMethod = OccludingObjectsInpaintMethod.EdgeByEdge


class JointRGBAndDepthInpainter:
    """Inpainter of the missing region of rgbd image.

    This method is based on the RGBD inpainting method described in 
    '3d ken burns effect from a single image'.
    """
    def __init__(self, option: JointRGBAndDepthOption):
        self.option = option

        if self.option.inpaint_method == OccludingObjectsInpaintMethod.U2NetPretrained:
            self.u2_mask_model = U2MaskModel()

    @classmethod
    def add_arguments(cls, parser: simple_parsing.ArgumentParser):
        parser.add_arguments(JointRGBAndDepthOption, dest="joint_rgbd")
    
    @classmethod
    def to_option(cls, arguments):
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

        if self.option.inpaint_method == OccludingObjectsInpaintMethod.EdgeByEdge:
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
        
        elif self.option.inpaint_method == OccludingObjectsInpaintMethod.OtsuMask:
            foreground_mask = create_foreground_mask(depth_image)
            inpaint_mask_image = np.ones(depth_image.shape)
            inpaint_mask_image[foreground_mask == 255] = 0
            self.__inpaint_with_inpaint_mask(rgb_image, depth_image, disparity_image, inpaint_mask_image)

        elif self.option.inpaint_method == OccludingObjectsInpaintMethod.U2NetPretrained:
            rgb_image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            mask = self.u2_mask_model.predict_mask(rgb_image_bgr)
            show_image_ui(rgb_image)
            mask_numpy = mask.squeeze(0).cpu().detach().numpy()
            st.write("## Gray Scale Mask")
            show_image_ui(mask_numpy, cmap=plt.cm.gray)
            mask_numpy_u16 = cv2.normalize(mask_numpy, None, 0, 2 ** 16-1, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            thresh, binary_mask = cv2.threshold(
                mask_numpy_u16,
                np.min(mask_numpy_u16),
                np.max(mask_numpy_u16),
                cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )
            mask_u8 = np.zeros_like(binary_mask, dtype=np.uint8)
            mask_u8[binary_mask == 2 ** 16 - 1] = 255
            st.write("## Binary Mask")
            show_image_ui(mask_u8, cmap=plt.cm.gray)
            box = create_roi_from_u8_mask(mask_u8, margin=20)

            inpaint_mask = np.ones_like(binary_mask, dtype=np.uint8)
            inpaint_mask[binary_mask > thresh] = 0
            inpaint_mask = cv2.erode(inpaint_mask, kernel=np.ones((3,3)), iterations=3)
            show_image_ui(inpaint_mask, cmap=plt.cm.gray)
            inpainted_rgb_image, inpainted_disparity_image = self.__inpaint_with_inpaint_mask(
                rgb_image_bgr[box], depth_image[box], disparity_image[box], inpaint_mask[box].astype(np.float32))

            result_rgb_image = rgb_image.copy()
            result_rgb_image[box][inpaint_mask[box] == 0] = (inpainted_rgb_image * 255)[inpaint_mask[box] == 0]
            result_disparity_image = disparity_image.copy()
            result_disparity_image[box][inpaint_mask[box] == 0] = inpainted_disparity_image[inpaint_mask[box] == 0]

            show_image_ui(result_rgb_image)
            show_image_ui(result_disparity_image)