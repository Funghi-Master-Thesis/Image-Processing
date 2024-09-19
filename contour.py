import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an color image in grayscale
image = cv2.imread('Data/IBT23253/1.jpeg',0)

# blur the image
bkz = 10
blurred = cv2.blur(image, (bkz, bkz), 0)

# thresholding
(T, thresh) = cv2.threshold(blurred, 150, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Morphological filters
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.erode(thresh, kernel, iterations=1)
#thresh = cv2.dilate(thresh, kernel, iterations=1)

# find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# filter contours
contour_list = []
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    if ((len(approx) > 8) & (area > 50000)):
        contour_list.append(contour)
print(len(contours))
print(len(contour_list))

# draw contours
thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
cv2.drawContours(thresh, contour_list, -1, (255,0,0), 3)

image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
cv2.drawContours(image, contour_list, -1, (255,0,0), 3)



plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(thresh)
plt.title('Original Image')
plt.axis('off')

# Canny Edge-detected Image
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.title('Canny Edge-detected Image')
plt.axis('off')

# Display the images
plt.show()
