import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Function to process user's image and calculate histogram
def process_user_image(image_path):
    # Load the image
    user_image = cv2.imread(image_path)
    return user_image

# Function to calculate and plot histogram with suggested thresholds
def plot_histogram_with_thresholds(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:  # Check if image is color
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Apply Gaussian smoothing to the histogram
    smoothed_hist = gaussian_filter1d(hist[:, 0], sigma=2)

    # Find peaks in the smoothed histogram
    peaks = np.where((smoothed_hist[1:-1] > smoothed_hist[:-2]) & (smoothed_hist[1:-1] > smoothed_hist[2:]))[0] + 1
    significant_peaks = peaks[smoothed_hist[peaks] > 20]  # Minimum threshold for peak significance

    # Find valleys in the smoothed histogram
    valleys = np.where((smoothed_hist[1:-1] < smoothed_hist[:-2]) & (smoothed_hist[1:-1] < smoothed_hist[2:]))[0] + 1
    all_valleys = valleys  # No height threshold applied

    # Plot the histogram with detected peaks and valleys
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_hist, color='black', label='Smoothed Histogram')
    plt.plot(hist, color='lightgray', label='Original Histogram', alpha=0.5)
    plt.xlim([0, 256])
    
    # Highlight significant peaks
    for peak in significant_peaks:
        plt.axvline(peak, color='blue', linestyle='--', label=f'Peak: {peak}')

    # Highlight all valleys
    for valley in all_valleys:
        plt.axvline(valley, color='orange', linestyle='--', label=f'Valley: {valley}')

    plt.title('Histogram of Pixel Intensities with Detected Peaks and Valleys')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()

    return significant_peaks, all_valleys, gray_image

# Function to apply Canny edge detection
def apply_canny_edge_detection(image, low_threshold, high_threshold):
    # Apply Canny edge detector
    edge_map = cv2.Canny(image, low_threshold, high_threshold)
    return edge_map

# Function to display edge maps one by one
def display_edge_maps(edge_maps, threshold_pairs, gray_image):
    for i, (edge_map, (low, high)) in enumerate(zip(edge_maps, threshold_pairs)):
        plt.figure(figsize=(10, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Edge map
        plt.subplot(1, 2, 2)
        plt.imshow(edge_map, cmap='gray')
        plt.title(f'Edges (Low: {low}, High: {high})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

     

# Example usage
image_path = "Data/RIS1_0_TL_20_preset/335.jpeg"
user_image = process_user_image(image_path)

# Generate histogram and extract significant peaks and valleys
significant_peaks, all_valleys, gray_image = plot_histogram_with_thresholds(user_image)

# Generate edge maps for combinations of adjacent peaks and valleys
edge_maps = []
threshold_pairs = []

# Iterate through significant peaks
for i, low_threshold in enumerate(significant_peaks):
    # Determine adjacent peaks and valleys
    adjacent_peaks = significant_peaks[max(0, i - 1): min(len(significant_peaks), i + 2)]
    adjacent_valleys = all_valleys[max(0, i - 1): min(len(all_valleys), i + 2)]
    
    # Generate thresholds from adjacent peaks and valleys
    for adj_peak in adjacent_peaks:
        if adj_peak != low_threshold:  # Avoid using the same peak for both thresholds
            high_threshold = min(255, adj_peak + 50)  # Simple offset for high threshold
            edge_map = apply_canny_edge_detection(gray_image, low_threshold, high_threshold)
            edge_maps.append(edge_map)
            threshold_pairs.append((low_threshold, high_threshold))
    
    for adj_valley in adjacent_valleys:
        high_threshold = min(255, adj_valley + 50)  # Simple offset for high threshold
        edge_map = apply_canny_edge_detection(gray_image, low_threshold, high_threshold)
        edge_maps.append(edge_map)
        threshold_pairs.append((low_threshold, high_threshold))

# Display edge maps one at a time
display_edge_maps(edge_maps, threshold_pairs, gray_image)
