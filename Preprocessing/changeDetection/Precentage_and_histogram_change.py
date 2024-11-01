import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.util import img_as_float
import os

def loadImage(image_path):
    # Load the image from the given path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Convert to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image
    # resized = cv2.resize(grey, (256, 256))
    # Convert to float format
    float_image = img_as_float(grey)
    return float_image

# Function to create synthetic test images
def create_test_images():
    # Create a black image
    # image1 = loadImage('D:/gitRepos/Image-Processing/Data/RIS1_0_TL_20_preset/150.jpeg')

    # # Create a slightly modified image (moving the circle)
    # image2 = loadImage('D:/gitRepos/Image-Processing/Data/RIS1_0_TL_20_preset/160.jpeg')

    image1 = loadImage('D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium/41217_10.jpeg')

    # Create a slightly modified image (moving the circle)
    image2 = loadImage('D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium/41217_79.jpeg')


    return image1, image2

# Function to compare two images and calculate the percentage change
def compare_images(image1, image2):
    # Ensure the images are in float format for accurate calculation
    image1_float = image1.astype(np.float32) / 255.0
    image2_float = image2.astype(np.float32) / 255.0
    
    # Compute absolute difference
    dif_img = np.abs(image2_float - image1_float)

    # Compute total absolute difference
    total_difference = np.sum(dif_img)

    # Calculate the total number of pixels
    total_pixels = dif_img.size

    # Calculate the percentage change
    percentage_change = (total_difference / (total_pixels * 1.0)) * 100  # Normalize by number of pixels

    return percentage_change, dif_img


def display_compare():
    # Create test images
    image1, image2 = create_test_images()

    # Compare the two images
    percentage_change, difference_image = compare_images(image1, image2)

    # Output the results
    print(f"Percentage Change: {percentage_change:.2f}%")
    print(percentage_change)

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference_image, cmap='hot')
    plt.title('Difference Image')
    plt.axis('off')

    plt.show()


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    score = cv2.compareHist(hist1, hist2, method)
    return score

def load_and_prepare_image(image_path, target_size=(256, 256), resize=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = gray_image
    if resize:
        resized_image = cv2.resize(gray_image, target_size)
        output = resized_image

    return output

def detect_and_plot_changes(image_folder, threshold=5, comparison_method=cv2.HISTCMP_CORREL):
    # Detect significant changes in images based on histogram comparison and plot the results.
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpeg")))
    if len(image_paths) < 2:
        print("Not enough images to compare.")
        return

    change_indices = [] 
    scores = []  
    significant_scores = [] 

    # Load and process the first image as the initial baseline
    baseline_image = load_and_prepare_image(image_paths[0])
    baseline_hist = calculate_histogram(baseline_image)

    for idx, image_path in enumerate(image_paths[1:], start=1):
        current_image = load_and_prepare_image(image_path)
        current_hist = calculate_histogram(current_image)
        
        # Compare histograms with the baseline
        score = compare_histograms(baseline_hist, current_hist, method=comparison_method)
        scores.append(score) 
        
        # Mark scores below threshold
        if score < 1.00:
            # baseline_hist = current_hist 
            change_indices.append(idx)
            significant_scores.append(score)
        baseline_hist = current_hist  
        
        print(f"Compared image {idx-1} to image {idx}: Score = {score:.3f}")

    # Plot the scores with significant changes highlighted
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(scores) + 1), scores, label="Similarity Score", color="blue")
    plt.scatter(change_indices, [scores[i - 1] for i in change_indices], color="red", marker='o', label="Significant Change")
    plt.axhline(y=threshold, color='green', linestyle='--', label="Threshold")
    
    plt.title("Image Change Detection Over Time")
    plt.xlabel("Image Index")
    plt.ylabel("Histogram Similarity Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nSignificant changes detected at the following image indices:", change_indices)
    return change_indices


Data_folder = 'D:/gitRepos/Image-Processing/Data/DataSetUniform/DataSet/Acremonium'

# display_compare()
# change_indices = detect_and_plot_changes(Data_folder)
