from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import os
import pandas as pd
import changeDetection.VarianceOverTime as vod
import shutil
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_folder_path = '.\Data\AllData'
output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetCutLast2Days')
info = pd.read_excel(r'.\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt = 'Preprocessing\\filters\\finished_cutlast2days.txt'
exclude = 'Preprocessing\\filters\\excludelast2d.txt'


lines = open(finished_ibt).read().splitlines()
exclude_l = open(exclude).read().splitlines()

lines.extend(exclude_l)
days = 2
image_count_last = days * 24 * 2

def apply_circular_mask(image, scale_factor=0.675):
    """Apply a circular mask to the image with an adjustable scale factor."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(center[0], center[1], width - center[0], height - center[1]) * scale_factor)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def cut_to_boundingbox(image):
    """Cut the image to the bounding box of the circle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    image = image[y:y+h, x:x+w]
    return image
def extract_and_save_petri_dishes(image_path, output_folder, resize_dim=None, ibt_number=''):
    """Extract each petri dish from the image and save them with labels."""
    # print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    height, width, _ = image.shape
    # print(f"Image dimensions: {width}x{height}")

    rows, cols = 3, 2
    petri_height = height // rows
    petri_width = width // cols

    for row in range(rows):
        for col in range(cols):
            x_start = col * petri_width
            y_start = row * petri_height
            x_end = x_start + petri_width
            y_end = y_start + petri_height

            petri_dish = image[y_start:y_end, x_start:x_end]
            petri_dish = apply_circular_mask(petri_dish)
            petri_dish = cut_to_boundingbox(petri_dish)
            # Resize if resize_dim is provided
           
            #If the array is empty continue to the next iteration
            if petri_dish.size == 0:
                continue
            petri_dish = cv2.resize(petri_dish, resize_dim)


            label = f"{ibt_number}_{os.path.splitext(os.path.basename(image_path))[0]}_row_{row+1}_col_{col+1}.jpg"
            output_path = os.path.join(output_folder, label)
            success = cv2.imwrite(output_path, petri_dish)
            if not success:
                print(f"Failed to save cropped image: {output_path}")

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

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    
    correct_folder = os.listdir(folder_path)[0]
    if correct_folder != "RIS1_0_TL_20_preset":
        print("Weird Folder, skipping" + correct_folder)
        continue
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    # get how many files that are in the folder
    file_count = len(os.listdir(folder_path))
    number = folder.split()[1]
    last_image_indices = range(file_count - image_count_last, file_count)
    last_image_indices = [ele for ele in last_image_indices if ele > 0]
    fungi_map[number] = last_image_indices

    

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    
    correct_folder = os.listdir(folder_path)[0]
    if correct_folder != "RIS1_0_TL_20_preset":
        print("Weird Folder, skipping" + correct_folder)
        continue
    folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    
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
    ibts = content + number + "\n"
    with open(finished_ibt, 'w') as file:
        file.write(ibts)
    print("Finished processing: " + folder)