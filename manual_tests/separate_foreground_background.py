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

rgb_image = cv2.imread("data/rgbd_inpaint_test/00027.jpg")

if not os.path.exists("data/rgbd_inpaint_test/depth.npy"):
    estimator = DPTDepthEstimator()
    mask_image = cv2.imread("data/rgbd_inpaint_test/00027.png")

    depth_image_container = estimator.estimate_and_get_depth_image_container(rgb_image)
    depth_image = depth_image_container.get_depth_image()

    np.save("data/rgbd_inpaint_test/depth", depth_image)
else:
    depth_image = np.load("data/rgbd_inpaint_test/depth.npy")


depth_median = np.percentile(depth_image, 20)
depth_mean_diff = np.median(np.abs(depth_image - depth_median))
depth_min = depth_median - depth_mean_diff
depth_max = depth_median + depth_mean_diff

depth_image_u16 = cv2.normalize(depth_image, None, 0, 2 ** 16, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

depth_image = depth_image.clip(depth_min, depth_max)

depth_image_u16 = cv2.normalize(depth_image, None, 0, 2 ** 16, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
depth_image = cv2.GaussianBlur(depth_image,(3,3),8)
edge_image = cv2.Laplacian(depth_image, cv2.CV_64F)

def zero_crossing_v2(LoG):
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
    return zeroCross


def zero_crossing_v3(LoG):
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
    sloop = maxLoG - minLoG
    zeroCross = np.logical_and(
        # This is the approximate solution to set t he minimum sloop.
        (sloop > np.percentile(sloop, 95)),
        np.logical_or(
            np.logical_and(minLoG < 0,  LoG > 0),
            np.logical_and(maxLoG > 0, LoG < 0))
    )
    gray = np.zeros(zeroCross.shape, dtype=np.uint8)
    gray[zeroCross] = 255
    st.write(zeroCross.dtype)
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2,2)))

show_image_ui(zero_crossing_v1(edge_image), cmap=plt.cm.gray)
show_image_ui(zero_crossing_v2(edge_image), cmap=plt.cm.gray)
show_image_ui(zero_crossing_v3(edge_image), cmap=plt.cm.gray)

edge_image = zero_crossing_v3(edge_image)
st.write(edge_image.dtype)
_, labels = cv2.connectedComponents(edge_image)

foreground_mask = np.zeros(edge_image.shape, np.uint8)

for label_id in np.unique(labels):
    if label_id == 0:
        continue
    one_edge_image = np.zeros(edge_image.shape, dtype=np.uint8)
    one_edge_image[labels == label_id] = 255

    dilated_edge_for_one_image = cv2.morphologyEx(
        one_edge_image, cv2.MORPH_DILATE, np.ones((3,3)), iterations=20)
    # show_image_ui(dilated_edge_for_one_image, cmap=plt.cm.gray)

    depth_in_the_region = depth_image_u16[dilated_edge_for_one_image == 255]
    ret_val, mask = cv2.threshold(
        depth_in_the_region,
        np.min(depth_in_the_region), 
        np.max(depth_in_the_region), 
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    foreground_image = np.zeros(edge_image.shape, dtype=np.uint8)
    foreground_slice=  np.bitwise_and(dilated_edge_for_one_image == 255, depth_image_u16 < ret_val)
    foreground_image[foreground_slice] = 255
    # show_image_ui(foreground_image, cmap=plt.cm.gray)

    foreground_mask[foreground_slice] =255

show_image_ui(labels)
show_image_ui(foreground_mask, cmap=plt.cm.gray)