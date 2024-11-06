
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.util import img_as_ubyte
from skimage import util
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from skimage.filters import sobel


def process_image_with_otsu_and_auto_canny():
    # Step 1: Load the image
    img_og = cv2.imread('Data/RIS1_0_TL_20_preset/329 copy.jpeg')
    img = util.invert(img_og)  # Inverting the image to enhance contrast
    ogh, ogw, _ = img.shape
    ogimg = img.copy()
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()

    # Step 2: Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    enhanced_img = clahe.apply(cimg)

    # Step 3: Pre-process the image (apply GaussianBlur to reduce noise)
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)

    # Step 4: Use Otsu's method to get the optimal threshold
    otsu_thresh_value, otsu_thresh_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Apply Canny Edge Detection using Otsu's threshold
    optimal_threshold = otsu_thresh_value / 2
    lower_thresh = int(max(0, 0.5 * optimal_threshold))
    upper_thresh = int(min(255, 1.5 * optimal_threshold))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    kernel = np.ones((8,8))
    # do a morphologic close
    edges_uint8 = cv2.morphologyEx(edges,cv2.MORPH_CLOSE, kernel)

    # Increase the radii range to capture all petri dishes
    hough_radii = np.arange(75, 100)
    hough_res = hough_circle(edges_uint8, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=6)

    # Step 8: Create a mask for all detected circles
    mask = np.zeros(ogimg.shape, np.uint8)
    for center_y, center_x, radius in zip(cy, cx, radii):
        # Adjust the radius scaling to ensure proper alignment with the original size
        cv2.circle(mask, (center_x * 10, center_y * 10), radius * 10, (255, 255, 255), -1)

    # Step 9: Apply the mask to the original image
    cv2.imwrite("mask.png", mask)
    mask2 = cv2.imread('mask.png', 0)
    res = cv2.bitwise_and(ogimg, ogimg, mask=mask2)
    cv2.imwrite("testmasked.jpeg", res)

    # Step 10: Display the results
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask2, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title('Masked Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image_path = 'Data/RIS1_0_TL_20_preset/329 copy.jpeg'  
    process_image_with_otsu_and_auto_canny()
