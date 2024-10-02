# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import math

# in_dir = "D:\gitRepos\Image-Processing\Data\RIS1_0_TL_20_preset/"
# # im_name = "335.jpeg"
# im_name = "335 - copy.jpeg"
# # Load the image

# image = cv2.imread(in_dir + im_name)

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian Blur to smooth the image
# blurred = cv2.GaussianBlur(gray, (3, 3), 5)

# # Use Canny edge detection to find edges
# edges = cv2.Canny(blurred, 5, 10)

# # Find contours from edges
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter the contours by size to detect petri dishes
# min_radius = 1200  # Minimum radius of a petri dish (adjust this if needed)
# detected_dishes = []  # Store each detected petri dish as a tuple (center, radius)

# for contour in contours:
#     # Fit a circle around the contour
#     (x, y), radius = cv2.minEnclosingCircle(contour)
#     center = (int(x), int(y))
#     radius = int(radius)
    
#     # Filter out small or large objects that aren't likely petri dishes
#     if radius > min_radius:
#         detected_dishes.append((center, radius))

# # Dynamically determine the grid size for plotting (e.g., 2x3 or 3x3 depending on the number of dishes)
# num_dishes = len(detected_dishes)
# grid_cols = math.ceil(math.sqrt(num_dishes))  # Number of columns based on total dishes
# grid_rows = math.ceil(num_dishes / grid_cols)  # Number of rows based on columns

# # Create a figure to display results
# plt.figure(figsize=(grid_cols * 4, grid_rows * 4))

# # Process each detected petri dish
# for i, (center, radius) in enumerate(detected_dishes):
#     # Create a mask for the current petri dish
#     mask = np.zeros_like(gray)
#     cv2.circle(mask, center, radius, 255, -1)

#     # Mask the original image to remove the background around the current petri dish
#     result = cv2.bitwise_and(image, image, mask=mask)

#     # Calculate bounding box for cropping
#     x_start = max(center[0] - radius, 0)  # Ensure not out of bounds
#     y_start = max(center[1] - radius, 0)
#     x_end = min(center[0] + radius, image.shape[1])
#     y_end = min(center[1] + radius, image.shape[0])

#     # Crop the image based on the bounding box
#     cropped = result[y_start:y_end, x_start:x_end]

#     # Display each petri dish
#     plt.subplot(grid_rows, grid_cols, i+1)  # Dynamic grid based on number of dishes
#     plt.title(f'Petri Dish {i+1}')
#     plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#     plt.axis('off')  # Hide axis for cleaner visualization

# # Show the plot with all detected petri dishes
# plt.tight_layout()
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

in_dir = "D:\gitRepos\Image-Processing\Data\RIS1_0_TL_20_preset/"
# im_name = "335.jpeg"
im_name = "335 - copy.jpeg"
# Load the image

image = cv2.imread(in_dir + im_name)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to smooth the image
blurred = cv2.GaussianBlur(gray, (3, 3), 5)

# Use Canny edge detection to find edges
edges = cv2.Canny(blurred, 5, 10)

# Find contours from edges
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours by size to detect petri dishes
min_radius = 1200  # Minimum radius of a petri dish (adjust this if needed)
detected_dishes = []  # Store each detected petri dish as a tuple (center, radius)

for contour in contours:
    # Fit a circle around the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Filter out small or large objects that aren't likely petri dishes
    if radius > min_radius:
        detected_dishes.append((center, radius))

# Process each detected petri dish individually
for i, (center, radius) in enumerate(detected_dishes):
    # Create a mask for the current petri dish
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)

    # Mask the original image to remove the background around the current petri dish
    result = cv2.bitwise_and(image, image, mask=mask)

    # Calculate bounding box for cropping
    x_start = max(center[0] - radius, 0)  # Ensure not out of bounds
    y_start = max(center[1] - radius, 0)
    x_end = min(center[0] + radius, image.shape[1])
    y_end = min(center[1] + radius, image.shape[0])

    # Crop the image based on the bounding box
    cropped = result[y_start:y_end, x_start:x_end]

    # Display each petri dish
    plt.figure(figsize=(4, 4))  # Create a new figure for each petri dish
    plt.title(f'Petri Dish {i+1}')
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for cleaner visualization
    plt.show()  # Display each image one by one
