import dataclasses

import numpy as np
from remimi.utils.depth import DPTDepthImageContainer, colorize2, get_foreground_background_edges
from remimi.utils.image import show_image_ui
from remimi.utils.file import get_model_file
import cv2
import streamlit as st

import torch

from remimi.inpaint.rgbd.mengli.networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net

@dataclasses.dataclass
class EdgeBasedRGBDInpainterOption:
    edge_inpaint_net_name: str = "mengli_edge_inpaint_net.pth"
    depth_inpaint_net_name: str = "mengli_depth_inpaint_net.pth"
    rgb_inpaint_net_name: str = "mengli_rgb_inpaint_net.pth"
    device: str = "cuda:0"

class EdgeBasedRGBDInpainter:
    """Inpainter of the missing region of rgbd image.

    This method is based on the RGBD inpainting method described in 
    '3D Photography using Context-aware Layered Depth Inpainting'.
    """
    def __init__(self, option: EdgeBasedRGBDInpainterOption = EdgeBasedRGBDInpainterOption()):
        self.option = option

        self.edge_inpaint_net = self._prepared_model(
            Inpaint_Edge_Net(init_weights=True),
            get_model_file(
                option.edge_inpaint_net_name, 
                "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth"
            )
        )
        self.depth_inpaint_net = self._prepared_model(
            Inpaint_Depth_Net(),
            get_model_file(
                option.depth_inpaint_net_name,
                "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth"
            )
        )
        self.rgb_inpainit_net = self._prepared_model(
            Inpaint_Color_Net(),
            get_model_file(
                option.rgb_inpaint_net_name,
                "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth"
            )
        )

    def _prepared_model(self, net, checkpoint_path):
        net_weight = torch.load(
            checkpoint_path, map_location=torch.device(self.option.device)
        )
        net.load_state_dict(net_weight, strict=True)
        net = net.to(self.option.device)
        net.eval()

        return net.to(self.option.device)

    def inpaint(self, rgb_image, depth_image_container: DPTDepthImageContainer):
        """Inpaint both rgb and depth images with mask_image.

        Arguments:
            - rgb_image: ccc
            - depth_image: bbb
            - mask_image: aaa
        Returns: 
            Inpainted image.
        """

        # import IPython; IPython.embed()

        depth_image = depth_image_container.get_depth_image()
        disparity_image = depth_image_container.get_inverse_depth_image()
    
        inpainted_depth_image = depth_image.copy()
        inpainted_rgb_image = rgb_image.copy()
        for i, (foreground_slice, background_slice, edge) in enumerate(get_foreground_background_edges(depth_image)):
            # # depth_image_u8 = colorize2(depth_image)
            # depth_image_u8 = cv2.normalize(depth_image, None, 0, 1000, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # edge_image = cv2.Laplacian(depth_image_u8, cv2.CV_64F)
            # # edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
            # edge_gray = np.zeros(edge_image.shape, dtype=np.uint8)
            # edge_gray[np.bitwise_or(edge_image < -0.5, edge_image > 0.5)] = 1
            # inpaint_mask_image = cv2.cvtColor(inpaint_mask_image, cv2.COLOR_BGR2GRAY)
            # inpaint_mask_image[inpaint_mask_image > 1] = 1
            # # inpaint_mask_image = cv2.bitwise_not(inpaint_mask_image)

            # input should be RGB.
            inpaint_mask_image = np.zeros(depth_image.shape)
            inpaint_mask_image[foreground_slice] = 1
            edge = edge / 255
            rgb_tensor = torch.FloatTensor(rgb_image /255.).permute(2, 0, 1)[None, ...].cuda()
            disparity_tensor = torch.FloatTensor(np.abs(disparity_image) / np.max(disparity_image))[None, None, ...].cuda()
            depth_tensor = torch.FloatTensor(depth_image)[None, None, ...].cuda()
            inpaint_mask_tensor = torch.FloatTensor(inpaint_mask_image)[None, None, ...].cuda()
            edge_tensor = torch.FloatTensor(edge)[None, None, ...].cuda()

            # kernel = np.ones((5,5),np.uint8)
            # dilated_inpaint_mask1 = cv2.dilate(inpaint_mask_image,kernel,iterations = 30)
            # context_mask = dilated_inpaint_mask1 - inpaint_mask_image
            # # context_mask = cv2.bitwise_not(context_mask)
            # context_mask[context_mask > 1] = 1
            # context_mask_tensor = torch.FloatTensor(context_mask)[None, None, ...].cuda()

            context_mask = np.zeros(depth_image.shape)
            context_mask[background_slice] = 1
            context_mask_tensor = torch.FloatTensor(context_mask)[None, None, ...].cuda()            

            edge_output = self.edge_inpaint_net.forward_3P(
                inpaint_mask_tensor, context_mask_tensor, 
                rgb_tensor, disparity_tensor, edge_tensor, cuda="cuda:0")

            depth_output = self.depth_inpaint_net.forward_3P(
                inpaint_mask_tensor, context_mask_tensor, depth_tensor, edge_output, cuda="cuda:0"
            )

            color_output = self.rgb_inpainit_net.forward_3P(
                inpaint_mask_tensor, context_mask_tensor, rgb_tensor, edge_output, cuda="cuda:0"
            )
            inpainted_depth = depth_output.squeeze(0).squeeze(0).cpu().numpy()
            inpainted_color = color_output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()

            inpainted_depth_image[inpaint_mask_image == 1] = inpainted_depth[inpaint_mask_image == 1]
            inpainted_rgb_image[inpaint_mask_image == 1] = inpainted_color[inpaint_mask_image == 1] * 255

            # st.write(f"{i}.")
            show_image_ui(inpaint_mask_image)
            show_image_ui(context_mask)
            show_image_ui(edge)
            show_image_ui(edge_output.squeeze(0).squeeze(0).cpu().numpy())
            show_image_ui(depth_output.squeeze(0).squeeze(0).cpu().numpy())
            show_image_ui(color_output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy())

        show_image_ui(inpainted_depth_image)
        show_image_ui(inpainted_rgb_image)