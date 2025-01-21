import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern,  canny, hog
from skimage.filters import sobel
from skimage.measure import regionprops, moments_hu
from skimage.transform import resize
from scipy.fftpack import fft2
from skimage import img_as_ubyte
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import scipy.ndimage as ndi  # Import ndi


# Texture Features: Local Binary Patterns (LBP)
def extract_lbp_features(image, P=8, R=1):
    gray_image = rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)  # Convert to 8-bit unsigned byte format
    lbp = local_binary_pattern(gray_image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

def extract_color_histogram(image, bins=256):
    hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

# Frequency Domain Features: Fourier Transform
def extract_fourier_features(image):
    gray_image = rgb2gray(image)
    f_transform = fft2(gray_image)
    f_transform = np.abs(f_transform)
    return f_transform

# Statistical Features: Mean and Variance
def extract_statistical_features(image):
    mean = np.mean(image, axis=(0, 1))
    variance = np.var(image, axis=(0, 1))
    return mean, variance



