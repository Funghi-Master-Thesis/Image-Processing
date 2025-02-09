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
output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetLast2Days')
info = pd.read_excel(r'.\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt = 'Preprocessing\\filters\\finished_last2days.txt'
exclude = 'Preprocessing\\filters\\exclude_last2days.txt'


lines = open(finished_ibt).read().splitlines()
exclude_l = open(exclude).read().splitlines()

lines.extend(exclude_l)
days = 2
image_count_last = days * 24 * 2



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
    mask_array = []
    if not os.path.exists(dataset_output):
        os.makedirs(dataset_output, exist_ok=True)
    print("Processing: " + number)
    mask = np.zeros((600, 400, 3), np.uint8)
    xrange = [100, 300, 500]
    
    for i in range(3):
        cv2.circle(mask, (100, xrange[i]), 75, (255, 255, 255), -1)
        cv2.circle(mask, (300, xrange[i]), 75, (255, 255, 255), -1)
    mask = cv2.resize(mask, (400,600))
    cv2.imwrite("mask.png", mask)
    mask = cv2.imread('mask.png',0)
    for indexes in indexes:
        image_path = folder_path + r'\\' + str(indexes) + '.jpeg'
        filename = os.path.basename(image_path)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            print("Processing file: " + filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            ogh, ogw, _ = img.shape
            ogimg = img.copy()
            img = cv2.resize(img, (400, 600))
            res = cv2.bitwise_and(img,img,mask = mask)
                # For comparison
            cv2.imwrite(dataset_output + '\\' + (number+"_"+filename), res)

    with open(finished_ibt, 'r') as file:
        content = file.read()
    ibts = content + number + "\n"
    with open(finished_ibt, 'w') as file:
        file.write(ibts)
    print("Finished processing: " + folder)