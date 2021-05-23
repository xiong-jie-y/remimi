import json
import os
from os.path import join
import subprocess

import numpy as np
from remimi.utils.binary_image import create_roi_from_u8_mask

from matplotlib import pyplot as plt
from remimi.utils.image import show_image_ui
from remimi.segmentation.u2net_wrapper import U2MaskModel, U2MaskModelOption
import click
import glob
import cv2
import streamlit as st
import tqdm
import shutil
import tempfile
import yaml

@click.command()
@click.option("--model-name")
@click.option("--dataset-definition-file")
@click.option("--output-path")
def main(model_name, dataset_definition_file, output_path):
    u2_mask_model = U2MaskModel(U2MaskModelOption(model_name=model_name))
    datasets = yaml.load(open(dataset_definition_file, "r"))

    categories = [
        {
            "id": 1,
            "name": "alchol sheet"
        },
        {
            "id": 2,
            "name": "ipad"
        },
    ]
    annotations_train = []
    annotations_val = []
    images_train = []
    images_val = []

    image_output_path = os.path.join(output_path, "images")
    os.makedirs(image_output_path, exist_ok=True)
    
    next_image_id = 0
    for dataset in datasets:
        with tempfile.TemporaryDirectory() as image_path:
            video_path = dataset['video_path']
            file_template = os.path.join(image_path, "$filename%06d.png")
            command = f"ffmpeg -i {video_path} -r 60/1 {file_template}"
            subprocess.run(command, shell=True, check=True)

            image_paths = sorted(list(glob.glob(join(image_path, "*.png"))) + list(glob.glob(join(image_path, "*.jpg"))))
            areas = []
            masks = []
            annotations = []
            images = []
            for image_path in tqdm.tqdm(image_paths):
                image_bgr = cv2.imread(image_path)
                mask = u2_mask_model.predict_mask(image_bgr)
                mask_numpy = mask.squeeze(0).cpu().detach().numpy()
                mask_u8 = u2_mask_model.convert_probability_mask_to_binary_mask(mask_numpy)
                roi = create_roi_from_u8_mask(mask_u8)

                contour_points = roi.contours[0].flatten().tolist()
                area = cv2.contourArea(np.array(roi.contours[0])) if len(roi.contours) != 0 else 0

                masks.append(mask_u8)

                if area == 0:
                    continue

                areas.append(area)

                # import IPython; IPython.embed()
                annotation = {
                    "segmentation": [contour_points],
                    # "area": float(max_x - min_x) * float(max_y - min_y),
                    "area": area,
                    "iscrowd": 0,
                    "image_id": next_image_id,
                    "bbox": [
                        float(roi.min_x), float(roi.min_y), 
                        float(roi.max_x - roi.min_x), float(roi.max_y - roi.min_y)],
                    "category_id": dataset["category_id"],
                    "id": next_image_id
                }
                image_name = os.path.basename(image_path)
                video_name = os.path.basename(video_path)
                new_image_name = f"{next_image_id}_{video_name}_{image_name}"
                image = {
                    "file_name": new_image_name,
                    "width": image_bgr.shape[1],
                    "height": image_bgr.shape[0],
                    "id": next_image_id,
                    "image_id": next_image_id
                }
                annotations.append(annotation)
                images.append(image)

                shutil.copy2(image_path, os.path.join(image_output_path, new_image_name))
                next_image_id += 1

            import scipy
            WINDOW_LENGTH = 5
            # area_moving_median = np.convolve(areas, np.ones(WINDOW_LENGTH), 'valid') / WINDOW_LENGTH
            area_moving_median = scipy.signal.medfilt(areas, WINDOW_LENGTH)
            st.write(area_moving_median)
            near_median_indices = []
            for i, area in enumerate(areas):
                if area_moving_median[i] * 0.7 < area and area < area_moving_median[i] * 1.3:
                    near_median_indices.append(i)

            # fig, ax = plt.subplots(1, 1)
            # ax.plot(list(range(0, len(areas))), areas)
            # st.pyplot(fig)

            # import IPython; IPython.embed()

            NUM_VAL_PER_NUM = 3
            num_added = 0

            for i in near_median_indices:
                if num_added % NUM_VAL_PER_NUM == 0:
                    annotations_val.append(annotations[i])
                    images_val.append(images[i])
                else:
                    annotations_train.append(annotations[i])
                    images_train.append(images[i])

                num_added += 1

    json.dump({
        "images": images_train,
        "annotations": annotations_train,
        "categories": categories,
        "info": {
            "description": "COCO 2017 Dataset"
        }
    }, open(os.path.join(output_path, "annotations.json"), "w"), indent=4)

    json.dump({
        "images": images_val,
        "annotations": annotations_val,
        "categories": categories,
        "info": {
            "description": "COCO 2017 Dataset"
        }
    }, open(os.path.join(output_path, "annotations_val.json"), "w"), indent=4)

if __name__ == "__main__":
    main()