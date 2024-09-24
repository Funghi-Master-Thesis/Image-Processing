#Import all necessery libraries
from matplotlib import pyplot as plt
import numpy as np
import cv2

#Read the image and perform threshold and get its height and weight
img = cv2.imread('Data/IBT23253/1.jpeg')
h, w = img.shape[:2]

# Transform to gray colorspace and blur the image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

# Make a fake rectangle arround the image that will seperate the main contour.
cv2.rectangle(blur, (0,0), (w,h), (255,255,255), 10)

# Perform Otsu threshold.
_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Create a mask for bitwise operation
mask = np.zeros((h, w), np.uint8)

# Search for contours and iterate over contours. Make threshold for size to
# eliminate others.
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for i in contours:
    cnt = cv2.contourArea(i)
    if 1000000 >cnt > 100000:
        cv2.drawContours(mask, [i],-1, 255, -1)


# Perform the bitwise operation.
res = cv2.bitwise_and(img, img, mask=mask)

# Display the result.
cv2.imwrite('mole_res.jpg', res)

plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(res)
plt.title('Original Image')
plt.axis('off')

# Canny Edge-detected Image


# Display the images
plt.show()