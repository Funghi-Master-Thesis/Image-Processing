
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
def check_overlap(x1, y1, r1, x2, y2, r2, w, h):
    # Check if the first circle is within the picture bounds
    if not (r1 <= x1 <= w - r1) or not (r1 <= y1 <= h - r1):
        return False  # Circle 1 is out of bounds

    # Check if the second circle is within the picture bounds
    if not (r2 <= x2 <= w - r2) or not (r2 <= y2 <= h - r2):
        return False  # Circle 2 is out of bounds

    # Calculate the distance between the centers of the circles
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Check if the circles overlap
    return distance < (r1 + r2)
folder_path = 'Data/IBT41495/'
output_folder = 'Data/Output/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        print("Processing file: " + filename)
        img = cv2.imread(file_path)
        ogh, ogw, _ = img.shape
        ogimg = img.copy()
        img = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
        cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        output=img.copy()
        low_threshold = 10
        high_threshold = 80
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

        # Example usage
        kernel = np.ones((3,3))
            # do a morphologic close
        edge_map = cv2.morphologyEx(edge_map,cv2.MORPH_CLOSE, kernel)


        hough_radii = np.arange(75, 100)
        hough_res = hough_circle(edge_map, hough_radii)

        # Select the most prominent 3 circles
        _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 100, 100, total_num_peaks=6)

        h, w, _ = img.shape
        mask = np.zeros((ogimg.shape), np.uint8)
        
        

# Initialize a list to store the detected circles that don't overlap
        valid_circles = []
        for center_y, center_x, radius in zip(cy, cx, radii):
            cv2.circle(mask, ((center_x*10), (center_y*10)), ((radius-20)*10), (255, 255, 255), -1)
        # Loop through detected circles
        # for center_y, center_x, radius in zip(cy, cx, radii):
        #     center_x_scaled, center_y_scaled, radius_scaled = center_x * 10, center_y * 10, radius * 10
        #     # Check overlap with previously added circles
        #     overlap = False
        #     for valid_circle in valid_circles:
        #         if check_overlap(center_x_scaled, center_y_scaled, radius_scaled, valid_circle[0], valid_circle[1], valid_circle[2], ogw, ogh):
        #             overlap = True
        #             break
            
        #     # If no overlap, add the circle to the valid list and draw it
        #     if not overlap:
        #         valid_circles.append((center_x_scaled, center_y_scaled, radius_scaled))
        #         cv2.circle(mask, (center_x_scaled, center_y_scaled), radius_scaled, (255, 255, 255), -1)
       
        cv2.imwrite("mask.png", mask)
        mask2 = cv2.imread('mask.png',0)
        # # Original Image
        res = cv2.bitwise_and(ogimg,ogimg,mask = mask2)
        cv2.imwrite("Data/Output/" + filename, res)


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