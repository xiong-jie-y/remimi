from os.path import join

from matplotlib import pyplot as plt
from remimi.utils.image import show_image_ui
from remimi.segmentation.u2net_wrapper import U2MaskModel, U2MaskModelOption
import click
import glob
import cv2
import streamlit as st

@click.command()
@click.option("--model-name")
@click.option("--image-path")
def main(model_name, image_path):
    u2_mask_model = U2MaskModel(U2MaskModelOption(model_name=model_name))
    image_paths = list(glob.glob(join(image_path, "*.png"))) + list(glob.glob(join(image_path, "*.jpg")))
    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        mask = u2_mask_model.predict_mask(image_bgr)
        show_image_ui(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mask_numpy = mask.squeeze(0).cpu().detach().numpy()
        st.write("## Gray Scale Mask")
        show_image_ui(mask_numpy, cmap=plt.cm.gray)
        mask_u8 = u2_mask_model.convert_probability_mask_to_binary_mask(mask_numpy)
        st.write("## Binary Mask")
        show_image_ui(mask_u8, cmap=plt.cm.gray)

if __name__ == "__main__":
    main()