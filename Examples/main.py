from scipy.ndimage import correlate
import numpy as np
from skimage import io, color, util, filters, exposure
import matplotlib.pyplot as plt
from skimage.filters import median
import cv2
from skimage.util import img_as_float, img_as_ubyte
from skimage.io import imread, imshow
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt


### exercise 1 
input_img = np.arange(25).reshape(5, 5)
weights = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]

# Correlate with reflection
res_img_reflection = correlate(input_img, weights)
print(res_img_reflection[3, 3])


### exercise 2 
# Correlate with constant border handling
res_img_constant = correlate(input_img, weights, mode="constant", cval=10)

print("Result image with reflection border handling:")
print(res_img_reflection)
print("\nResult image with constant border handling:")
print(res_img_constant)

### Exercise 3
def preprocess_image(image_filename, noise_type=None):
    # Define the directory containing the images
    in_dir = "data/"

    # Read the image
    img = io.imread(in_dir + image_filename)

    # Convert image to grayscale if it's not already
    if len(img.shape) > 2:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img

    # Convert image to floating-point representation
    img_float = img_as_float(img_gray)

    return img_float

Gaussian = "Gaussian.png"
SaltPepper = "SaltPepper.png"
car = "car.png"
donald1 = "donald_1.png"
ElbowCTSlice = "ElbowCTSlice.png"

funghi_prefix = "IBT23253/"

funghi = funghi_prefix + "60.jpeg"

gaussian_img = preprocess_image(Gaussian)
salt_pepper_img = preprocess_image(SaltPepper)
car_img = preprocess_image(car)
donald1_img = preprocess_image(donald1)
ElbowCTSlice_img = preprocess_image(ElbowCTSlice)
funghi_img = preprocess_image(funghi)



def meanFiltered(image, size):
    io.imshow(image)
    plt.title('Guassian')
    io.show()

    # Two dimensional filter filled with 1
    weights = np.ones([size, size])
    # Normalize weights
    weights = weights / np.sum(weights)

    filtered_mean = correlate(image, weights, mode="constant", cval=10)

    io.imshow(filtered_mean)
    plt.title('filtered Guassian')
    io.show()


### Exercise 4
def medianFiltering(image, size):
    footprint = np.ones([size, size])
    med_img = median(image, footprint)

    filtered_median = correlate(med_img, weights, mode="constant", cval=10)

    io.imshow(filtered_median)
    plt.title('median filtered Guassian')
    io.show()

### Exercise 5
def differentFilteringSizes(image):
    # Define filter sizes
    filter_sizes = [3, 5, 10, 20]

    # Iterate over filter sizes
    for size in filter_sizes:
        # Create median filter footprint
        footprint = np.ones((size, size))
        
        # Apply median filter
        med_img = median(image, footprint)

        # Apply mean filter
        mean_filtered_image = correlate(image, np.ones((size, size)) / (size * size), mode='constant', cval=0.0)
        
        # Apply median filter again on the previously median filtered image
        median_filtered_image = median(med_img, np.ones((size, size)))

        # Display mean filtered image
        plt.imshow(mean_filtered_image, cmap='gray')
        plt.title(f'Mean Filtered Image (Size={size})')
        plt.show()

        # Display median filtered image
        plt.imshow(median_filtered_image, cmap='gray')
        plt.title(f'Median Filtered Image (Size={size})')
        plt.show()

### Exercise 6
def gaussFiltering(image):
    sigma = 1
    gauss_img = gaussian(image, sigma)

    #gauss_filtered_image = gaussian(gauss_img, np.ones((size, size)))

    # Display median filtered image
    plt.imshow(gauss_img, cmap='gray')
    plt.title('gauss Filtered Image')
    plt.show()

def prewitt(image):
    # Apply horizontal Prewitt filter
    gradient_h = filters.prewitt_h(image)

    # Apply vertical Prewitt filter
    gradient_v = filters.prewitt_v(image)

    gradient_img = filters.prewitt(image)

    plt.imshow(gradient_h, cmap='gray')
    plt.title('gradient_h')
    plt.show()

    plt.imshow(gradient_v, cmap='gray')
    plt.title('gradient_v')
    plt.show()

def edgeDetection(image):

    # Step 1: Compute the gradients in the filtered image using a Prewitt filter
    gradient_img = filters.prewitt(image)
    
    # Apply Otsu's thresholding to compute binary image
    threshold = filters.threshold_otsu(gradient_img)
    binary_img = gradient_img >= threshold
    
    # Convert binary image to unsigned byte
    proc_img = img_as_ubyte(binary_img)

    io.imshow(proc_img, cmap='gray')
    io.show()



# ### Exercise 3
# meanFiltered(gaussian_img, 5)

# ### Exercise 4
# medianFiltering(gaussian_img, 5)

# ### Exercise 5
# differentFilteringSizes(gaussian_img)

### Exercise 6
# gaussFiltering(gaussian_img)

# ### Exercise 7
# meanFiltered(car_img, 15)
# medianFiltering(car_img, 15)
# gaussFiltering(car_img, 15)

# ### Exercise 8
#prewitt(donald1_img)

### Exercise 9

### Exercise 10
# edgeDetection(ElbowCTSlice_img)
# edgeDetection(funghi_img)
