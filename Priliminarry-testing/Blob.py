# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, feature, filters, measure

# # Function to apply the Kirsch operator
# def kirsch_filter(image):
#     kirsch_kernels = [
#         np.array([[5, 5, 5],
#                   [-3, 0, -3],
#                   [-3, -3, -3]]),
#         np.array([[5, 5, -3],
#                   [5, 0, -3],
#                   [-3, -3, -3]]),
#         np.array([[5, -3, -3],
#                   [5, 0, -3],
#                   [5, -3, -3]]),
#         np.array([[-3, -3, -3],
#                   [5, 0, -3],
#                   [5, 5, -3]]),
#         np.array([[-3, -3, -3],
#                   [-3, 0, 5],
#                   [-3, 5, 5]]),
#         np.array([[-3, -3, 5],
#                   [-3, 0, 5],
#                   [-3, -3, -3]]),
#         np.array([[-3, 5, 5],
#                   [-3, 0, -3],
#                   [-3, -3, -3]]),
#         np.array([[5, -3, -3],
#                   [-3, 0, -3],
#                   [-3, 5, -3]])
#     ]
#     response = np.zeros(image.shape)
#     for kernel in kirsch_kernels:
#         filtered = cv2.filter2D(image, -1, kernel)
#         response = np.maximum(response, filtered)  # Get the maximum response
#     return response

# in_dir = "D:\gitRepos\Image-Processing\Data\RIS1_0_TL_20_preset/"
# im_name = "335.jpeg"
# # Load the image

# image = cv2.imread(in_dir + im_name)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Kirsch filter to find edges
# gradient_image = kirsch_filter(gray_image)

# # Threshold the gradient image to create a binary mask
# _, binary_mask = cv2.threshold(gradient_image, 100, 255, cv2.THRESH_BINARY)

# # Find contours in the binary mask
# contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create a mask to cut out the background
# final_mask = np.zeros_like(gray_image)

# # Filter contours based on area and draw them on the mask
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 750:  # Minimum area to filter out noise
#         cv2.drawContours(final_mask, [contour], -1, 255, -1)

# # Apply the mask to the original image to cut out the background
# masked_image = cv2.bitwise_and(image, image, mask=final_mask)

# # Show results
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(gray_image, cmap='gray')
# plt.title('Grayscale Image')

# plt.subplot(1, 3, 2)
# plt.imshow(gradient_image, cmap='gray')
# plt.title('Gradient Image (Kirsch Filter)')

# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
# plt.title('Masked Image')
# plt.axis('off')
# plt.show()

# # GLCM for texture analysis
# glcm = feature.greycomatrix(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY), 
#                              [1], [0],  symmetric=True, normed=True)

# # Calculate texture features
# contrast = feature.greycoprops(glcm, 'contrast')
# dissimilarity = feature.greycoprops(glcm, 'dissimilarity')
# homogeneity = feature.greycoprops(glcm, 'homogeneity')

# print("GLCM Texture Features:")
# print(f"Contrast: {contrast[0][0]}")
# print(f"Dissimilarity: {dissimilarity[0][0]}")
# print(f"Homogeneity: {homogeneity[0][0]}")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

# Function to apply the Kirsch operator
def kirsch_filter(image):
    kirsch_kernels = [
        np.array([[5, 5, 5],
                  [-3, 0, -3],
                  [-3, -3, -3]]),
        np.array([[5, 5, -3],
                  [5, 0, -3],
                  [-3, -3, -3]]),
        np.array([[5, -3, -3],
                  [5, 0, -3],
                  [5, -3, -3]]),
        np.array([[-3, -3, -3],
                  [5, 0, -3],
                  [5, 5, -3]]),
        np.array([[-3, -3, -3],
                  [-3, 0, 5],
                  [-3, 5, 5]]),
        np.array([[-3, -3, 5],
                  [-3, 0, 5],
                  [-3, -3, -3]]),
        np.array([[-3, 5, 5],
                  [-3, 0, -3],
                  [-3, -3, -3]]),
        np.array([[5, -3, -3],
                  [-3, 0, -3],
                  [-3, 5, -3]])
    ]
    response = np.zeros(image.shape)
    for kernel in kirsch_kernels:
        filtered = cv2.filter2D(image, -1, kernel)
        response = np.maximum(response, filtered)  # Get the maximum response
    return response

in_dir = "D:\\gitRepos\\Image-Processing\\Data\\RIS1_0_TL_20_preset\\"
im_name = "335.jpeg"

# Load the image
image = cv2.imread(in_dir + im_name)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Kirsch filter to find edges
gradient_image = kirsch_filter(gray_image)

# Threshold the gradient image to create a binary mask
_, binary_mask = cv2.threshold(gradient_image, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for petri dishes (initially all zeros)
final_mask = np.zeros_like(gray_image)

# Filter contours based on area and draw them on the mask
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 600:  # Minimum area to filter out noise
        cv2.drawContours(final_mask, [contour], -1, 255, -1)

# Invert the mask to keep the petri dishes and remove the background
inverted_mask = cv2.bitwise_not(final_mask)

# Apply the inverted mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)

# Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 3, 2)
plt.imshow(gradient_image, cmap='gray')
plt.title('Gradient Image (Kirsch Filter)')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.title('Masked Image with Petri Dishes')
plt.axis('off')
plt.show()

# GLCM for texture analysis
glcm = feature.greycomatrix(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY), 
                             [1], [0], symmetric=True, normed=True)

# Calculate texture features
contrast = feature.greycoprops(glcm, 'contrast')
dissimilarity = feature.greycoprops(glcm, 'dissimilarity')
homogeneity = feature.greycoprops(glcm, 'homogeneity')

print("GLCM Texture Features:")
print(f"Contrast: {contrast[0][0]}")
print(f"Dissimilarity: {dissimilarity[0][0]}")
print(f"Homogeneity: {homogeneity[0][0]}")
