from remimi.utils.depth import clip_depth_to_foreground, colorize2, create_foreground_mask, create_foreground_mask_v2, remove_outlier
from matplotlib import pyplot as plt
import streamlit as st
import os
import cv2
import numpy as np
import torch
from remimi.monodepth.dpt import DPTDepthEstimator
from remimi.utils.image import show_image_ui

def zero_crossing_v1(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
 
 
            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
 
            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)
 
    return z_c_image

rgb_image = cv2.imread("data/examples/person_portrait.jpg")

estimator = DPTDepthEstimator()
depth_image_container = estimator.estimate_and_get_depth_image_container(rgb_image)
depth_image = depth_image_container.get_depth_image()

clipped_depth = clip_depth_to_foreground(depth_image)
clipped_depth_image_u16 = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
show_image_ui(clipped_depth_image_u16, cmap=plt.cm.inferno)

np.save("data/rgbd_inpaint_test/depth", depth_image)

st.markdown("## Occlusion Mask for Looking Glass")
show_image_ui(create_foreground_mask(depth_image), cmap=plt.cm.gray)

st.markdown("## Background Removal")
show_image_ui(rgb_image[:, :, ::-1])
mask = create_foreground_mask_v2(depth_image)
show_image_ui(mask, cmap=plt.cm.gray)
rgb_image_masked = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
show_image_ui(rgb_image_masked[:, :, ::-1])