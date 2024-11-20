from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import os
import pandas as pd
import changeDetection.VarianceOverTime as vod
import shutil
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from individual_cut import extract_and_save_petri_dishes

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_folder_path = 'D:\gitRepos\Image-Processing\Data\RIS1_0_TL_20_preset'
output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetCut')
info = pd.read_excel(r'D:\gitRepos\Image-Processing\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt_detail = 'Preprocessing\\finishedibt.txt'
finished_ibt = 'Preprocessing\\finishedibtjustnumber.txt'

lines = open(finished_ibt).read().splitlines()

def extract_significant_images(image_folder, baseline_image_path, area_threshold, visualize=False):
    """Process a folder of images to detect significant changes from a baseline image."""
    baseline = vod.load_and_process(baseline_image_path)
    significant_images = []

    image_files = sorted(image_folder, key=vod.natural_sort_key)
    baseline_index = image_files.index(baseline_image_path) if baseline_image_path in image_files else -1
    image_files = image_files[baseline_index + 1:]  # Exclude images before the baseline
    for image_path in image_files:
        image = vod.load_and_process(image_path)
        diff = vod.compute_difference(baseline, image)
        optimal_thresholded_value = vod.gaussian_otsu_threshold(image)
        thresholded_diff = vod.apply_threshold(diff, optimal_thresholded_value)

        if vod.detect_significant_change(thresholded_diff, area_threshold):
            image_number = vod.natural_sort_key(image_path)
            significant_images.append(image_number[1])

            if visualize:
                cv2.imshow("Difference", diff)
                cv2.imshow("Thresholded Difference", thresholded_diff)
                cv2.waitKey(10)

    return significant_images

def get_significant_image_array(folder_path):
    image_folder = folder_path
    baseline_image_path = find_suitable_baseline(image_folder)

    significant_images = extract_significant_images(
        image_folder,
        baseline_image_path,
        area_threshold=30,
        visualize=True
    )
    return significant_images

def find_suitable_baseline(folder_path):
    images = [f for f in folder_path if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in folder {folder_path}.")

    images.sort(key=vod.natural_sort_key)

    for image_path in images:
        print(f"Processing image {image_path}")
        dew_present = vod.test_differences(image_path, identified_blobs=100)
        if not dew_present:
            return image_path

    raise ValueError("No suitable baseline image found." + image_path)

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    ibtmap = {}

    dataset_output = os.path.join(base_path, output_folder, folder)
    if not os.path.exists(dataset_output):
        os.mkdir(dataset_output)
    for filename in os.listdir(folder_path):
        filename_without_extension = os.path.splitext(filename)[0]
        result = filename_without_extension.split("_")[0]
        if result not in ibtmap:
            ibtmap[result] = []
        ibtmap[result].append(os.path.join(folder_path, filename))

    for ibt in ibtmap:
        if ibt == "RIS1_0_TL_20_preset":
            significant_images = get_significant_image_array(ibtmap[ibt])
            for image_path in significant_images:
                extract_and_save_petri_dishes(image_path, dataset_output, resize_dim=(224, 224))
            cv2.waitKey(0)
    print("Finished processing: " + folder)