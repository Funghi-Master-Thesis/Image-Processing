import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = "data/IBT23253/70.jpeg"
image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply thresholding to separate the petri dishes from the background
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Step 4: Find contours of the petri dishes
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty mask where the background is black
mask = np.zeros_like(image)

# Step 5: Fill the mask with the contours of the petri dishes
cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Step 6: Apply the mask to the original image
masked_image = cv2.bitwise_and(image, mask)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Image with Mask Applied")
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

plt.show()
