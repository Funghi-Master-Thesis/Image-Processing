import cv2
import numpy as np
import os
import re
import cv2 as cv
from matplotlib import pyplot as plt

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
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def gaussian_otsu_threshold(image):
    """Apply an optimized threshold based on gaussian blur and otsu to the difference image to detect significant changes."""
    # enhanced_img = load_and_process(image)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    otsu_thresh_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh_value

def detect_significant_change(thresholded_diff, area_threshold):
    """Determine if the area of the changed region is significant based on a pixel count threshold."""
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_area = sum(cv2.contourArea(contour) for contour in contours)
    return significant_area > area_threshold

def natural_sort_key(filename):
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

def adaptive_thresholding_th3(image):
    """Apply adaptive thresholding using the THRESH_BINARY method."""
    img_blurred = cv.medianBlur(image, 5)
    th3 = cv.adaptiveThreshold(img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return th3

def detect_circles(image, min_radius=40, max_radius=100):
    """Detect circles in an image using Hough Circle Transform."""
    circles = cv.HoughCircles(
        image,
        cv.HOUGH_GRADIENT,
        dp=1.2,             # Inverse ratio of accumulator resolution to image resolution
        minDist=100,         # Minimum distance between detected circles
        param1=100,          # Higher threshold for Canny edge detector
        param2=30,          # Accumulator threshold for circle detection, increase to detect fewer circles
        minRadius=min_radius,
        maxRadius=max_radius
    )
    return circles

def test_differences(image_path, vizualize=False, identified_blobs=50):
    """ Test for dew presence in the image """
    # Read the image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "File could not be read, check with os.path.exists()"

    dewPressent = False

    # Detect circles in the image
    circles = detect_circles(img)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Create a mask for all the petri dish areas
        combined_mask = np.zeros_like(img)

        # Draw slightly smaller circles on the combined mask to exclude the rim
        for (x, y, r) in circles:
            inner_radius = r - 5  # Make the inner circle slightly smaller to avoid the rim
            if inner_radius > 0:  # Ensure the inner radius is positive
                cv.circle(combined_mask, (x, y), inner_radius, 255, thickness=-1)

        # Apply the combined mask to the original image 
        masked_img = cv.bitwise_and(img, img, mask=combined_mask)

        # Apply adaptive thresholding to the masked image (to highlight potential dew inside the circles)
        th3 = adaptive_thresholding_th3(masked_img) 
        th3_inv = cv.bitwise_not(th3)

        # Count non-zero pixels within the mask (to check for dew)
        non_zero_count = cv.countNonZero(th3_inv & combined_mask)
        print(f"Total non-zero pixel count in all circles: {non_zero_count}")

        # Blob analysis on the thresholded image (detecting the dew spots)
        num_labels, labels_im = cv.connectedComponents(th3_inv)

        print(f"Number of blobs detected inside the petridishes: {num_labels - 1}")  

        if num_labels > identified_blobs:
            dewPressent = True

        if vizualize:
            # Display results
            plt.figure(figsize=(12, 6))

            # Display the original image
            plt.subplot(1, 3, 1), plt.imshow(img, 'gray')
            plt.title("Original Image")
            plt.axis("off")

            # Display the masked image (inside circles only)
            plt.subplot(1, 3, 2), plt.imshow(masked_img, 'gray')
            plt.title("Masked Image (Inside Circles Only)")
            plt.axis("off")

            # Display the thresholded (inverted) image showing detected features (dew)
            plt.subplot(1, 3, 3), plt.imshow(th3_inv, 'gray')
            plt.title("Adaptive Thresholding (Inverted)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            # Optionally visualize the blobs
            output_img = np.zeros_like(img)
            output_img = cv.applyColorMap(labels_im.astype(np.uint8), cv.COLORMAP_JET)
            
            plt.imshow(output_img)
            plt.title("Blob Detection Results")
            plt.axis("off")
            plt.show()

    else:
        print("No circles detected.")

    return dewPressent


def find_suitable_baseline(folder_path):
    """
    Find the first suitable baseline image in the folder.
    
    Parameters:
    - folder_path: Path to the folder containing images.
    
    Returns:
    - str: Path to the first suitable baseline image.
    """
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in folder {folder_path}.")
    
    # Sort images using natural_sort_key
    images.sort(key=natural_sort_key)
    
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        print(f"Processing image {image_path}")
        
        # Use test_differences to check for dew presence
        dew_present = test_differences(image_path)
        
        # If no dew is present, return this image as the baseline
        if not dew_present:
            return image_path
    
    raise ValueError("No suitable baseline image found.")

def extract_significant_images(image_folder, baseline_image_path, area_threshold, visualize=False):
    """Process a folder of images to detect significant changes from a baseline image."""

    baseline = load_and_process(baseline_image_path)
    # baseline = find_suitable_baseline(image_folder)
    significant_images = []

    image_files = sorted(os.listdir(image_folder), key=natural_sort_key)
    baseline_image_name = os.path.basename(baseline_image_path)
    baseline_index = image_files.index(baseline_image_name) if baseline_image_name in image_files else -1
    image_files = image_files[baseline_index + 1:]  # Exclude images before the baseline
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = load_and_process(image_path)

        # Compute the difference and apply threshold
        diff = compute_difference(baseline, image)
        optimal_thresholded_value = gaussian_otsu_threshold(image)
        thresholded_diff = apply_threshold(diff, optimal_thresholded_value)
        # thresholded_diff = gaussian_otsu_threshold(diff)

        # Check if there's significant change
        if detect_significant_change(thresholded_diff, area_threshold):
            image_number = natural_sort_key(image_name)  # Use extracted number for indexing
            significant_images.append(image_number)
            print(f"Processing image {image_number}")

            # Optional: Show the detected changes
            if visualize:
                cv2.imshow("Difference", diff)
                cv2.imshow("Thresholded Difference", thresholded_diff)
                cv2.waitKey(10)  # Adjust this for speed of visualization

    
    return significant_images


# image_folder = 'E:/fredd/Uni/Thesis/Image-Processing/Data/DataSetUniform/DataSet/Penicillium-arizonense'
# # image_folder = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium-strictum'
# baseline_image_path = find_suitable_baseline(image_folder)

# significant_images = extract_significant_images(
#     image_folder,
#     baseline_image_path,
#     area_threshold=30,   # Adjust based on observed growth sizes
#     visualize=True # change for vizual output
# )

# print(f"Indices of images with significant growth: {significant_images}")

