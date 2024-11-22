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
data_folder_path = '.\Data\AllData'
output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetCut')
significant_dataset = os.path.join(base_path, 'Data', 'Output', 'DataSetSig')
info = pd.read_excel(r'.\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt_detail = 'Preprocessing\\filters\\finishedibt.txt'
finished_ibt = 'Preprocessing\\filters\\finishedibtjustnumber.txt'

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

def get_significant_image_array(folder_path, image_path):
    image_folder = folder_path
    baseline_image_path = find_suitable_baseline(image_folder, image_path)

    significant_images = extract_significant_images(
        image_folder,
        baseline_image_path,
        area_threshold=30,
        visualize=True
    )
    return significant_images

def find_suitable_baseline(folder_path, baseline_image_path):
    images = [f for f in folder_path if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in folder {folder_path}.")

    images.sort(key=vod.natural_sort_key)

    for image_path in images:
        print(f"Processing image {image_path}")
        dew_present = vod.test_differences(baseline_image_path + r'/' + image_path, identified_blobs=100)
        if not dew_present:
            return image_path

    raise ValueError("No suitable baseline image found." + image_path)

fungi_map = {}

for folder in os.listdir(significant_dataset):
    for file in os.listdir(os.path.join(significant_dataset, folder)):
        file = file.replace('.jpeg', '')
        index = int(file.split('_')[1])
        ibt = file.split('_')[0]
        if ibt not in fungi_map:
            fungi_map[ibt] = []
        fungi_map[ibt].append(index)
    

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    
    correct_folder = os.listdir(folder_path)[0]
    if correct_folder != "RIS1_0_TL_20_preset":
        print("Weird Folder, skipping" + correct_folder)
        continue
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    
    ibtmap = {}
    test1 = ibtinfo.index[ibtinfo.str.contains(folder)]
    if(test1.size == 0):
        continue
    test = test1[0]
    genus = info.iloc[test]['genus']
    species = info.iloc[test]['species']
    if isinstance(species, str) != True:
        species = ""
    species = species.replace("\"", "")
    
    if species == "":
        fungi_class = genus.strip()
    else:
        fungi_class = genus.strip() + '-' + species.strip()   
    number = folder.split()[1]
    if number in lines:
        print("Already processed " + number + ", skipping!")
        continue
    dataset_output = os.path.join(base_path, output_folder, fungi_class)
    os.makedirs(dataset_output, exist_ok=True)
    print(f"Output folder created: {output_folder}")
    print(f"Output folder path: {output_folder}")

    if number not in fungi_map:
        print("No significant images found for " + number + ", skipping!")
        continue
    indexes = fungi_map[number]
    
    print("Processing images in folder: " + folder)
    for indexes in indexes:
        image_path = folder_path + r'/' + str(indexes) + '.jpeg'
        extract_and_save_petri_dishes(image_path, dataset_output, resize_dim=(224, 224), ibt_number=number)

    with open(finished_ibt, 'r') as file:
        content = file.read()
    ibts = content + ibt + "\n"
    with open(finished_ibt, 'w') as file:
        file.write(ibts)
    print("Finished processing: " + folder)