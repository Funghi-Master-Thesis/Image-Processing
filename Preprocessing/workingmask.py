
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


img = cv2.imread('Data/RIS1_0_TL_20_preset/329 copy.jpeg')
ogh, ogw, _ = img.shape
ogimg = img.copy()
img = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
output=img.copy()
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

# Example usage
low_threshold = 80
high_threshold = 115
edge_map = canny_edge_detector(low_threshold, high_threshold)
kernel = np.ones((3,3))
    # do a morphologic close
edge_map = cv2.morphologyEx(edge_map,cv2.MORPH_CLOSE, kernel)


hough_radii = np.arange(75, 100)
hough_res = hough_circle(edge_map, hough_radii)

# Select the most prominent 3 circles
_, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 50, 50, total_num_peaks=6)

h, w, _ = img.shape
mask = np.zeros((ogimg.shape), np.uint8)
for center_y, center_x, radius in zip(cy, cx, radii):
    cv2.circle(mask, ((center_x*10), (center_y*10)), ((radius-20)*10), (255, 255, 255), -1)

cv2.imwrite("mask.png", mask)

mask2 = cv2.imread('mask.png',0)
# # Original Image
res = cv2.bitwise_and(ogimg,ogimg,mask = mask2)

def crop_petri_dishes(masked_image, original_height, original_width, centers_x, centers_y, radii):
    cropped_images = []
    for i, (center_y, center_x, radius) in enumerate(zip(centers_y, centers_x, radii)):
        # Calculate the bounding box coordinates
        x1 = max((center_x * 10) - (radius * 10), 0)  # left
        y1 = max((center_y * 10) - (radius * 10), 0)  # top
        x2 = min((center_x * 10) + (radius * 10), original_width)  # right
        y2 = min((center_y * 10) + (radius * 10), original_height)  # bottom

        # Crop the image around the detected circle from the masked image
        cropped_masked_img = masked_image[int(y1):int(y2), int(x1):int(x2)]
        
        # Save the cropped masked image
        output_path = f"cropped_masked_petri_dish_{i+1}.png"
        cv2.imwrite(output_path, cropped_masked_img)
        cropped_images.append(output_path)
        print(f"Cropped masked petri dish saved as {output_path}")
    
    return cropped_images

# Call the cropping function
# cropped_images = crop_petri_dishes(res, ogh, ogw, cx, cy, radii)


cv2.imwrite("testmasked.jpeg", res)

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the images
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB))
plt.title('Edge')
plt.axis('off')

# Display the images
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
plt.title('Mask')
plt.axis('off')

# Canny Edge-detected Image

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Masked image')
plt.axis('off')
# Display the images
plt.show()