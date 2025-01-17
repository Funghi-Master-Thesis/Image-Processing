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
    denoised = rank.median(img_as_ubyte(gray_image), disk(2))
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(2))
    labels = watershed(gradient, markers)
    return labels
def filter_segments(labels, imageRaw, area_threshold):
    unique_labels = np.unique(labels)
    segments = []
    filtered_labels = np.zeros_like(labels)
    average_colors = []
    most_occuring_colors = []
    all_segment_pixels = []
    
    for label in unique_labels:
        if label == 0:
            continue  # Skip the background label
        mask = labels == label
        area = np.sum(mask)
        if area >= area_threshold:
            segment_pixels = imageRaw[mask]
            average_color = np.mean(segment_pixels, axis=0)
            segment_pixels_tuple = [tuple(pixel) for pixel in segment_pixels]
            most_common_color = Counter(segment_pixels_tuple).most_common(1)[0][0]
            if most_common_color != (0, 0, 0):
                filtered_labels[mask] = label
                segments.append((label, area))
                average_colors.append((label, average_color))
                most_occuring_colors.append((label, most_common_color))
                all_segment_pixels.extend(segment_pixels)
    
    total_average_color = np.mean(all_segment_pixels, axis=0)
    return segments, average_colors, most_occuring_colors, total_average_color, filtered_labels
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
    base_path = 'E:\\fredd\\Uni\\Thesis\\Datasets\\AllDatasets\\DataSetCutLast2Days'
    image_path = base_path + '\\Aspergillus-arachidicola\\27190_218_row_2_col_2.jpg'
    background_colors = [np.array([40, 75, 165]), np.array([200, 200, 200])]  # orange , purple
    tolerance = 20  
    area_threshold = 150  
    imageRaw = read_image(image_path)
    fungal_mask = create_background_mask(imageRaw, background_colors, tolerance)
    masked_image = apply_mask(imageRaw, fungal_mask)
    labels = segment_image(masked_image)
    segments, average_colors, most_occuring_colors, total_average_color, filtered_labels = filter_segments(labels, imageRaw, area_threshold)
    print(f"Number of segments: {len(segments)}")
    print("Segments (label, area, average color, most occurring color):")
    for segment, avg_color, most_color in zip(segments, average_colors, most_occuring_colors):
        print(f"Label: {segment[0]}, Area: {segment[1]}, Average Color: {avg_color}, Most Occurring Color: {most_color}")
    print(f"Total Average Color from all segments: {total_average_color}")
    display_results(rgb2gray(masked_image), fungal_mask, labels, filtered_labels, True
    )
if __name__ == "__main__":
    main()