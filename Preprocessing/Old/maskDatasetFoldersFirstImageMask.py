
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

output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSet')
# info = pd.read_excel(r'C:\Users\Bruger\Documents\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')
info = pd.read_excel(r'E:\fredd\Uni\Thesis\Image-Processing\Data\DataDescription.xlsx')

ibtinfo = info['IBT number']
finished_ibt_detail = 'Preprocessing\\finishedibt.txt'
finished_ibt = 'Preprocessing\\finishedibtjustnumber.txt'

lines = open(finished_ibt).read().splitlines()


def canny_edge_detector(low_threshold, high_threshold):
    # Read the image using OpenCV

    # Apply Gaussian smoothing
    blurred_image = cv2.GaussianBlur(cimg, (5, 5), 0)

    # Compute gradients using Sobel operators
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Non-maximum suppression
    non_max_suppressed = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Edge tracking by hysteresis
    edge_map = cv2.Canny(blurred_image, low_threshold, high_threshold)


    return edge_map

def optimal_threshold():
    # Step 4: Use Otsu's method to get the optimal threshold
    blurred_image = cv2.GaussianBlur(cimg, (5, 5), 0)
    otsu_thresh_value, otsu_thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Apply Canny Edge Detection using Otsu's threshold
    optimal_threshold = otsu_thresh_value / 2
    lower_thresh = int(max(0, 0.5 * optimal_threshold))
    upper_thresh = int(min(255, 1.5 * optimal_threshold))

    return lower_thresh, upper_thresh


def scaling_factor(image):
   
    # we know that in an image with six petridishes, each dish roughley takes up 1/6 of the iamge.
    # therfore we can divide the image size by six and get the scaling factor for for the radii.
    width, hight = image.shape
    scaling = (1/6)
    width = width * scaling
    hight = hight * scaling

    return int(width), int(hight)


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
        os.mkdir(dataset_output)
    print("Processing: " + number)
    first_image = True
    first_image_mask = None
    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            print("Processing file: " + filename)
            img = cv2.imread(file_path)
            if img is None:
                continue
            ogh, ogw, _ = img.shape
            ogimg = img.copy()
            img = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
            cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            output=img.copy()
            if first_image:
                low, high = optimal_threshold()

                edge_map = canny_edge_detector(low, high)
                kernel = np.ones((3,3))
                    # do a morphologic close
                edge_map = cv2.morphologyEx(edge_map,cv2.MORPH_CLOSE, kernel)

                # 
                maxRad, minRad = scaling_factor(cimg)

                hough_radii = np.arange(minRad, maxRad)
                hough_res = hough_circle(edge_map, hough_radii)

                # Select the most prominent 3 circles
                _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 100, 100, total_num_peaks=6)

                h, w, _ = img.shape
                mask = np.zeros((ogimg.shape), np.uint8)

                for center_y, center_x, radius in zip(cy, cx, radii):
                    reduction = 0
                    if(radius > maxRad - 20):
                        reduction = radius - (maxRad - 20)
                    cv2.circle(mask, ((center_x*10), (center_y*10)), ((radius-reduction)*10), (255, 255, 255), -1)
                    
                # For comparison
                resized_mask = cv2.resize(mask, (0, 0), fx = 0.1, fy = 0.1)
                first_image_mask = mask
                mask_array.append(resized_mask)
                
                cv2.imwrite("mask.png", mask)
                mask2 = cv2.imread('mask.png',0)
                # # Original Image
                res = cv2.bitwise_and(ogimg,ogimg,mask = mask2)
                res = cv2.resize(res, (0, 0), fx = 0.1, fy = 0.1)
                
                cv2.imwrite(dataset_output + '\\' + (number+"_"+filename), res)
                first_image = False
            else:
                # For comparison
                resized = cv2.resize(first_image_mask, (0, 0), fx = 0.1, fy = 0.1)
                mask_array.append(resized)
                
                mask2 = cv2.imread('mask.png',0)
                mask2 = cv2.resize(mask2, (ogw, ogh))
                res = cv2.bitwise_and(ogimg,ogimg,mask = mask2)
                res = cv2.resize(res, (0, 0), fx = 0.1, fy = 0.1)
                
                cv2.imwrite(dataset_output + '\\' + (number+"_"+filename), res)
                
    merged = np.zeros((400, 600, 3), np.uint8)
    merged = cv2.resize(merged, (400, 600))
    for mask in mask_array:
        test = mask
        h1, w1, _ = test.shape
        if h1 != 600 or w1 != 400:
            test = cv2.resize(test, (400, 600))
        merged = cv2.bitwise_or(merged, test)
        
    mask_array.clear()
    cv2.imwrite(dataset_output + '\\' + ("00"+number+"_"+"MergedMask.png"), merged)
    with open(finished_ibt_detail, 'r') as file:
        content = file.read()
    new = content +number + " - " + fungi_class + "\n"
    with open(finished_ibt_detail, 'w') as file:
        file.write(new)
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