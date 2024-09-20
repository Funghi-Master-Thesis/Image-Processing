import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("Data/SmallerTest/1petri.png")
original = image.copy()

blur = cv2.GaussianBlur(image, (3,3), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
circle_mask = np.zeros(original.shape, dtype=np.uint8) 
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 200) 

# Convert the (x, y) coordinates and radius of the circles to integers
circles = np.round(circles[0, :]).astype("int")
circle_ratio = 0.85
cv2.imshow('circle_mask', circle_mask)
# Loop over the (x, y) coordinates and radius of the circles
cv2.waitKey(0) 