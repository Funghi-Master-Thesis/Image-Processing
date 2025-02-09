
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# data_folder_path = 'D:\\AllData'
data_folder_path = 'E:\\fredd\\Uni\\Thesis\\Image-Processing\\Data\\AllData'

output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetUniform')
# info = pd.read_excel(r'C:\Users\Bruger\Documents\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')
info = pd.read_excel(r'E:\fredd\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt = 'Preprocessing\\filters\\finished_uniform.txt'

lines = open(finished_ibt).read().splitlines()

for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
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
    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            print("Processing file: " + filename)
            img = cv2.imread(file_path)
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
    
    print("Finished with: " + folder)


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