
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import changeDetection.VarianceOverTime as vod
import shutil
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# data_folder_path = 'D:\\AllData'
data_folder_path = 'E:\\fredd\\Uni\\Thesis\\Image-Processing\\Data\\DataSetNoSpecies\\DataSetNoSpecies'

output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSetNoSpeciesLast2Days')
# info = pd.read_excel(r'C:\Users\Bruger\Documents\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')
info = pd.read_excel(r'E:\fredd\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt = 'Preprocessing\\filters\\last2daysfinished.txt'
exclude_list = 'Preprocessing\\filters\\exclude.txt'
days = 2
image_count_last = days * 24 * 2


lines = open(finished_ibt).read().splitlines()
# exclude_list = open(exclude_list).read().splitlines()
# lines.extend(exclude_list)
for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    ibtmap = {}
    
    dataset_output = os.path.join(base_path, output_folder, folder)
    if not os.path.exists(dataset_output):
        os.mkdir(dataset_output)
    for filename in os.listdir(folder_path):
        filename_without_extension = os.path.splitext(filename)[0]
        result = filename_without_extension.split("_")[0]
        if result not in ibtmap:
            ibtmap[result] = []
        ibtmap[result].append(os.path.join(folder_path, filename))
        # shutil.copy(file_path, dataset_output)
    # significant_image_indices = vod.get_significant_image_array(folder_path)
            
    for ibt in ibtmap:
        if ibt in lines:
            print("Already processed " + ibt + ", skipping!")
            continue
        ibt_list = sorted(ibtmap[ibt], key=vod.natural_sort_key)
        ibt_string = "IBT " + ibt
        ibt_info = info[info['IBT number'] == ibt_string]
    
        if ibt_info.empty:
            print(f"No data found for IBT number: {ibt_info}")
            continue
        
        # Assuming there is a column 'Image Count' that contains the count of images
        image_count = ibt_info['image no'].sum()
        last_image_indices = range(image_count - image_count_last, image_count)
        for image in last_image_indices:
            if ibt_list.__len__() <= image:
                break
            shutil.copy(ibt_list[image-1], dataset_output)
        with open(finished_ibt, 'r') as file:
            content = file.read()
        ibts = content + ibt + "\n"
        with open(finished_ibt, 'w') as file:
            file.write(ibts)
    print("Finished processing: " + folder)

            
                
                
            

    
                





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