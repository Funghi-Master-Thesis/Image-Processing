
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


in_dir = "D:\gitRepos\Image-Processing\Data\RIS1_0_TL_20_preset/"
im_name = "335.jpeg"
# Load the image

img = cv2.imread(in_dir + im_name)

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
low_threshold = 5
high_threshold = 50
edge_map = canny_edge_detector(low_threshold, high_threshold)
kernel = np.ones((3,3))
# do a morphologic close
edge_map = cv2.morphologyEx(edge_map,cv2.MORPH_CLOSE, kernel)

hough_radii = np.arange(75, 100)
hough_res = hough_circle(edge_map, hough_radii)



def detect_similar_circles(edge_map, size_threshold=25, min_distance_between_circles=75):
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        edge_map,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=75,   # Reduce minDist for better circle spacing detection
        param1=50,    # Edge detection sensitivity
        param2=20,    # Circle detection sensitivity (lower values detect more circles)
        minRadius=30, # Set appropriate minRadius based on your image
        maxRadius=125  # Set appropriate maxRadius based on your image
    )

    detected_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integer

        for circle in circles[0]:
            is_similar_size = all(abs(circle[2] - c[2]) <= size_threshold for c in detected_circles)
            is_far_enough = all(np.linalg.norm(np.array([circle[0], circle[1]]) - np.array([c[0], c[1]])) >= min_distance_between_circles for c in detected_circles)
            if is_similar_size and is_far_enough:
                detected_circles.append(circle)

        # Ensure no overlap between detected circles
        non_overlapping_circles = []
        for i, circle1 in enumerate(detected_circles):
            overlapping = False
            for j, circle2 in enumerate(detected_circles):
                if i != j:
                    # Calculate the Euclidean distance between circle centers
                    distance_between_centers = np.linalg.norm(np.array([circle1[0], circle1[1]]) - np.array([circle2[0], circle2[1]]))
                    # Check if the circles overlap
                    if distance_between_centers < (circle1[2] + circle2[2]):
                        overlapping = True
                        # Optionally keep the larger circle if they overlap
                        if circle1[2] < circle2[2]:
                            break  # Break the loop and discard the smaller circle
            if not overlapping:
                non_overlapping_circles.append(circle1)

        return non_overlapping_circles

# Detect similar circles with updated parameters
detected_circles = detect_similar_circles(edge_map, size_threshold=25, min_distance_between_circles=75)

# Total number of detected dishes
num_dishes = len(detected_circles)
print(f"Total number of detected dishes: {num_dishes}")

# Optional: Draw the detected circles on the original image for visualization
output_image = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

for circle in detected_circles:
    center = (circle[0], circle[1])  # Center coordinates
    radius = circle[2]                # Radius
    # Draw the circle outline
    cv2.circle(output_image, center, radius, (0, 255, 0), 2)
    # Draw the center of the circle
    cv2.circle(output_image, center, 2, (0, 0, 255), 3)

# Show the output image
cv2.imshow('Detected Petri Dishes', output_image)








# Select the most prominent circles
_, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 50, 50, total_num_peaks=6)

h, w, _ = img.shape
mask = np.zeros((ogimg.shape), np.uint8)
for center_y, center_x, radius in zip(cy, cx, radii):
    cv2.circle(mask, ((center_x*10), (center_y*10)), ((radius-20)*10), (255, 255, 255), -1)

cv2.imwrite("mask.png", mask)

mask2 = cv2.imread('mask.png',0)
# # Original Image
res = cv2.bitwise_and(ogimg,ogimg,mask = mask2)
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

