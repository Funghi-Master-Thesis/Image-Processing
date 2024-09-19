import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny_edge_detector(image_path, low_threshold, high_threshold):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian smoothing
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

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
image_path = 'data/IBT23253/70.jpeg'
low_threshold = 5
high_threshold = 100
edge_map = canny_edge_detector(image_path, low_threshold, high_threshold)

# Display the original and Canny edge-detected images
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Plotting the images using matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')

# Canny Edge-detected Image
plt.subplot(1, 2, 2)
plt.imshow(edge_map, cmap='gray')
plt.title('Canny Edge-detected Image')
plt.axis('off')

# Display the images
plt.show()