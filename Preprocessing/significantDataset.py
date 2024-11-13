
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import changeDetection.VarianceOverTime as vod
import shutil
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# data_folder_path = 'D:\\AllData'
data_folder_path = 'E:\\fredd\\Uni\\Thesis\\Image-Processing\\Data\\DataSetUniform\\DataSet'

output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetSig')
# info = pd.read_excel(r'C:\Users\Bruger\Documents\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')
info = pd.read_excel(r'E:\fredd\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt_detail = 'Preprocessing\\finishedibt.txt'
finished_ibt = 'Preprocessing\\finishedibtjustnumber.txt'

lines = open(finished_ibt).read().splitlines()


def extract_significant_images(image_folder, baseline_image_path, area_threshold,   visualize=False):
    """Process a folder of images to detect significant changes from a baseline image."""

    baseline = vod.load_and_process(baseline_image_path)
    # baseline = find_suitable_baseline(image_folder)
    significant_images = []

    image_files = sorted(image_folder, key=vod.natural_sort_key)

    for image_path in image_files:
        image = vod.load_and_process(image_path)

        # Compute the difference and apply threshold
        diff = vod.compute_difference(baseline, image)
        optimal_thresholded_value = vod.gaussian_otsu_threshold(image)
        thresholded_diff = vod.apply_threshold(diff, optimal_thresholded_value)
        # thresholded_diff = gaussian_otsu_threshold(diff)

        # Check if there's significant change
        if vod.detect_significant_change(thresholded_diff, area_threshold):
            image_number = vod.natural_sort_key(image_path)  # Use extracted number for indexing
            significant_images.append(image_number)
            print(f"Processing image {image_number}")

            # Optional: Show the detected changes
            if visualize:
                cv2.imshow("Difference", diff)
                cv2.imshow("Thresholded Difference", thresholded_diff)
                cv2.waitKey(10)  # Adjust this for speed of visualization

    return significant_images

def get_significant_image_array(folder_path):
    image_folder = folder_path
    baseline_image_path = find_suitable_baseline(image_folder)

    significant_images = extract_significant_images(
        image_folder,
        baseline_image_path,
        area_threshold=30,   # Adjust based on observed growth sizes
        visualize=False# change for vizual output
    )
    return significant_images

def find_suitable_baseline(folder_path):
    """
    Find the first suitable baseline image in the folder.
    
    Parameters:
    - folder_path: Path to the folder containing images.
    
    Returns:
    - str: Path to the first suitable baseline image.
    """
    images = [f for f in folder_path if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in folder {folder_path}.")
    
    # Sort images using natural_sort_key
    images.sort(key=vod.natural_sort_key)
    
    for image_path in images:
        print(f"Processing image {image_path}")
        
        # Use test_differences to check for dew presence
        dew_present = vod.test_differences(image_path)
        
        # If no dew is present, return this image as the baseline
        if not dew_present:
            return image_path
    
    raise ValueError("No suitable baseline image found.")

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    ibtmap = {}
    # dataset_output = os.path.join(base_path, output_folder, folder)
    # if not os.path.exists(dataset_output):
    #     os.mkdir(dataset_output)
    for filename in os.listdir(folder_path):
        filename_without_extension = os.path.splitext(filename)[0]
        result = filename_without_extension.split("_")[0]
        if result not in ibtmap:
            ibtmap[result] = []
        ibtmap[result].append(os.path.join(folder_path, filename))
        # shutil.copy(file_path, dataset_output)
    # significant_image_indices = vod.get_significant_image_array(folder_path)
    for ibt in ibtmap:
        dataset_output = os.path.join(base_path, output_folder, ibt)
        if not os.path.exists(dataset_output):
            os.mkdir(dataset_output)
            
    for ibt in ibtmap:
        if ibt == "41217":
            test = get_significant_image_array(ibtmap[ibt])
            print(test)
            print("hello")
        # dataset_output = os.path.join(base_path, output_folder, ibt)
        # for image in ibtmap[ibt]:
        #     file_path = os.path.join(folder_path, image)
        #     output_path = os.path.join(dataset_output, image)
        #     shutil.copy(file_path, output_path)
        #     break
            
                
                
            

    
                





# plt.subplot(1, 4, 1)
# plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# # Display the images
# plt.subplot(1, 4, 2)
# plt.imshow(cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB))
# plt.title('Edge')
# plt.axis('off')

# # Display the images
# plt.subplot(1, 4, 3)
# plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
# plt.title('Mask')
# plt.axis('off')

# # Canny Edge-detected Image

# plt.subplot(1, 4, 4)
# plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
# plt.title('Masked image')
# plt.axis('off')
# # Display the images
# plt.show()