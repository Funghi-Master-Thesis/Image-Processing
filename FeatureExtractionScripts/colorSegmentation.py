import os
import pandas as pd
import csv
from scipy import ndimage as ndi
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
                all_segment_pixels.extend(segment_pixels)
    
    total_average_color = np.mean(all_segment_pixels, axis=0)
    return segments, average_colors, total_average_color, filtered_labels

def build_annotation_dataframe(image_location, annot_location, output_csv_name):
    """Builds dataframe and csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    image_location: image directory path, e.g. r'.\data\train'
    annot_location: annotation directory path
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    class_lst = os.listdir(image_location)  # returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort()  # IMPORTANT
    with open(os.path.join(annot_location, output_csv_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index', 'average_color'])  # create column names
        for class_name in class_lst:
            class_path = os.path.join(image_location, class_name)  # concatenates various path components with exactly one directory separator (‘/’) except the last path component.
            file_list = os.listdir(class_path)  # get list of files in class folder
            print(f"Working on: {class_name}")
            i = 0
            file_count = len(file_list)
            for file_name in file_list:
                
                file_path = os.path.join(image_location, class_name, file_name)  # concatenate class folder dir, class name and file name
                imageRaw = read_image(file_path)
                background_colors = [np.array([40, 75, 165]), np.array([200, 200, 200])]  # Replace with the actual background colors
                tolerance = 20  # Adjust the tolerance as needed
                area_threshold = 150  # Adjust this value as needed
                fungal_mask = create_background_mask(imageRaw, background_colors, tolerance)
                masked_image = apply_mask(imageRaw, fungal_mask)
                labels = segment_image(masked_image)
                segments, average_colors, total_average_color, filtered_labels = filter_segments(labels, imageRaw, area_threshold)
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name), total_average_color])
                percentage_done = (int((i / file_count)*100))
                i = i + 1
                if(percentage_done%20 == 0):
                    print(f"{percentage_done}% completed")
                
    return pd.read_csv(os.path.join(annot_location, output_csv_name))

def main():
    image_location = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days'
    output_csv_name = 'ColorFeature.csv'
    df = build_annotation_dataframe(image_location, image_location, output_csv_name)
    print(df)

if __name__ == "__main__":
    main()