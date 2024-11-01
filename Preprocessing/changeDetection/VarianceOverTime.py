import cv2
import numpy as np
import os
import re

def load_and_process(image_path):
    """Load and preprocess an image by converting it to grayscale and applying contrast enhancement."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)  # Histogram equalization for contrast enhancement
    return enhanced

def compute_difference(image1, image2):
    """Compute the absolute difference between images."""
    diff = cv2.absdiff(image1, image2)
    return diff

# not currently in use
def apply_threshold(diff, threshold_value):
    """Apply a threshold to the difference image to detect significant changes."""
    _, thresholded = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def gaussian_otsu_threshold(image):
    """Apply an optimized threshold based on gaussian blur and otsu to the difference image to detect significant changes."""
    enhanced_img = load_and_process(image)
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    otsu_thresh_value, otsu_thresh_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    optimal_threshold = otsu_thresh_value / 2

    return otsu_thresh_value

def detect_significant_change(thresholded_diff, area_threshold):
    """Determine if the area of the changed region is significant based on a pixel count threshold."""
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_area = sum(cv2.contourArea(contour) for contour in contours)
    return significant_area > area_threshold

def natural_sort_key(filename):
    # only used for visual purpose, and sorting. 
    """
    Extracts IBT number and image number from the filename.
    Sorts first by type, then by image number.
    Example:
      '41217_1' -> (41217, 1)
    """
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        type_number = int(match.group(1))  
        image_number = int(match.group(2))  
        return (type_number, image_number)
    return (float('inf'), float('inf'))  


def extract_significant_images(image_folder, baseline_image_path, threshold_value, area_threshold, visualize=False):
    """Process a folder of images to detect significant changes from a baseline image."""
    baseline = load_and_process(baseline_image_path)
    significant_images = []

    image_files = sorted(os.listdir(image_folder), key=natural_sort_key)

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = load_and_process(image_path)

        # Compute the difference and apply threshold
        diff = compute_difference(baseline, image)
        thresholded_diff = apply_threshold(diff, threshold_value)

        # Check if there's significant change
        if detect_significant_change(thresholded_diff, area_threshold):
            image_number = natural_sort_key(image_name)  # Use extracted number for indexing
            significant_images.append(image_number)
            print(f"Processing image {image_number}")

            # Optional: Show the detected changes
            if visualize:
                cv2.imshow("Difference", diff)
                cv2.imshow("Thresholded Difference", thresholded_diff)
                cv2.waitKey(100)  # Adjust this for speed of visualization

    
    return significant_images

# Baseline has to reflect the base case, and can not have major imperfections such as dew on the dish
baseline_image_path = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium/41217_5.jpeg'
image_folder = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium'

# baseline_image_path = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Aspergillus-flavus/36710_0.jpeg'
# image_folder = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Aspergillus-flavus'



optimalThreshold= gaussian_otsu_threshold(baseline_image_path)
print(f"optimal Theshold found {optimalThreshold}")

significant_images = extract_significant_images(
    image_folder,
    baseline_image_path,
    threshold_value=optimalThreshold,  # Adjust based on lighting conditions or choose optimal threshold
    area_threshold=20,   # Adjust based on observed growth sizes
    visualize=False # change for vizual output
)

# Output the paths of images with significant changes
# print("Images with significant growth detected:")
# for img in significant_images:
#     print(img)

print(f"Indices of images with significant growth: {significant_images}")
