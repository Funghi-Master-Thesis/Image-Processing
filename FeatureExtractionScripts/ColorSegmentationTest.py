from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import numpy as np
import cv2
from collections import Counter
from skimage.feature import local_binary_pattern

# Import functions from FeatureExtraction.py
from FeatureExtraction import (
    extract_lbp_features,
    extract_color_histogram,
    extract_fourier_features,
    extract_statistical_features,
)


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path.")
    return img_as_ubyte(image)

def create_background_mask(image, background_colors, tolerance):
    background_mask = np.zeros(image.shape[:2], dtype=bool)
    for background_color in background_colors:
        lower_bound = background_color - tolerance
        upper_bound = background_color + tolerance
        mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=2)
        background_mask |= mask
    return ~background_mask

def apply_mask(image, mask):
    return image * np.dstack([mask] * 3)

def segment_image(image):
    gray_image = rgb2gray(image)
    denoised = rank.median(img_as_ubyte(gray_image), disk(5))
    markers = rank.gradient(denoised, disk(3)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(2))
    labels = watershed(gradient, markers)
    return labels

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def filter_segments(labels, imageRaw, area_threshold, black_pixel_threshold=0.1):
    unique_labels = np.unique(labels)
    segments = []
    filtered_labels = np.zeros_like(labels)
    average_colors = []
    all_segment_pixels = []
    total_area = 0
    weighted_sum = np.zeros(3)
    
    for label in unique_labels:
        if label == 0:
            continue  # Skip the background label
        mask = labels == label
        area = np.sum(mask)
        if area >= area_threshold:
            segment_pixels = imageRaw[mask]
            black_pixels = np.sum(np.all(segment_pixels == [0, 0, 0], axis=1))
            black_pixel_ratio = black_pixels / area
            if black_pixel_ratio > black_pixel_threshold:
                continue  # Skip segments with too many black pixels
            average_color = np.mean(segment_pixels, axis=0)
            segment_pixels_tuple = [tuple(pixel) for pixel in segment_pixels]
            most_common_color = Counter(segment_pixels_tuple).most_common(1)[0][0]
            if most_common_color != (0, 0, 0):
                filtered_labels[mask] = label
                segments.append((label, area))
                average_colors.append((label, average_color))
                all_segment_pixels.extend(segment_pixels)
                weighted_sum += average_color * area
                total_area += area
    
    total_average_color = weighted_sum / total_area if total_area > 0 else np.zeros(3)
    total_average_color_hex = rgb_to_hex(*total_average_color)
    return segments, average_colors, total_average_color_hex, filtered_labels

def display_results(gray_image, fungal_mask, markers, filtered_labels, vizualize=False):
    if vizualize:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
        ax = ax.ravel()
        ax[0].imshow(gray_image, cmap=plt.cm.gray)
        ax[0].set_title("Original")
        ax[1].imshow(fungal_mask, cmap=plt.cm.gray)
        ax[1].set_title("Fungal Mask")
        ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
        ax[2].set_title("Markers")
        ax[3].imshow(gray_image, cmap=plt.cm.gray)
        ax[3].imshow(filtered_labels, cmap=plt.cm.nipy_spectral, alpha=0.5)
        ax[3].set_title("Filtered Segments")
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        plt.show()

def main():
    image_path = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/Aspergillus-aculeatinus\\32384_240_row_3_col_1.jpg'
    background_colors = [np.array([75, 40, 145]), np.array([42, 131, 214])]  # purple, orange
    tolerance = 20  
    area_threshold = 100  
    imageRaw = read_image(image_path)
    fungal_mask = create_background_mask(imageRaw, background_colors, tolerance)
    masked_image = apply_mask(imageRaw, fungal_mask)
    labels = segment_image(masked_image)
    
    segments, average_colors, total_average_color_hex, filtered_labels = filter_segments(labels, imageRaw, area_threshold)
    lbp_features = extract_lbp_features(imageRaw)
    fourier_features = extract_fourier_features(imageRaw)
    mean, variance = extract_statistical_features(imageRaw)
    
    print("Segments:", segments)
    print("Average Colors:", average_colors)
    print("Total Average Color (Hex):", total_average_color_hex)
    print("LBP Features:", lbp_features)
    print("Fourier Features:", fourier_features)
    print("Mean:", mean)
    print("Variance:", variance)

    

if __name__ == "__main__":
    main()