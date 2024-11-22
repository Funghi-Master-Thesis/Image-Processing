from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
 


img = cv2.imread('Data/IBT23253/1.jpeg')
img = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
output=img.copy()
circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,50,75,
                            param1=50,param2=100,minRadius=85,maxRadius=150)
 
circles = np.uint16(np.around(circles))
if circles is not None:
     circles = np.round(circles[0, :]).astype("int")
     for (x, y, r) in circles:
    
         cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    
plt.figure(figsize=(10, 5))

# Original Image


# Canny Edge-detected Image
plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Canny Edge-detected Image')
plt.axis('off')

# Display the images
plt.show()
