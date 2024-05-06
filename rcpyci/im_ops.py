import os
import numpy as np
from PIL import Image


### save/load image operations
def save_image(image, path, clip=True, scale=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if clip:
        image = np.clip(image, 0, 1)
    if scale:
        image = image * 255
    Image.fromarray(image.astype(np.uint8)).save(path)

def read_image(filename, grayscale=True, maximize_contrast=True):
    img = Image.open(filename)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    if maximize_contrast:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.asarray(img)

### image processing operators
def apply_mask(ci, mask):
    if isinstance(mask, str):
        mask_matrix = read_image(mask, grayscale=True)
    elif isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_matrix = mask
    else:
        raise ValueError("The mask argument is neither a path to file nor a 2D matrix!")
    masked_ci = np.ma.masked_where(mask_matrix == 0, ci)
    return masked_ci

def apply_constant_scaling(ci, constant):
    scaled = (ci + constant) / (2 * constant)
    if np.any((scaled > 1.0) | (scaled < 0)):
        print("Chosen constant value for constant scaling made noise "
              "of classification image exceed possible intensity range "
              "of pixels (<0 or >1). Choose a lower value, or clipping "
              "will occur.")
    return scaled

def apply_matched_scaling(ci, base):
    min_base = np.min(base)
    max_base = np.max(base)
    min_ci = np.min(ci[~np.isnan(ci)])
    max_ci = np.max(ci[~np.isnan(ci)])
    scaled = min_base + ((max_base - min_base) * (ci - min_ci) / (max_ci - min_ci))
    return scaled

def apply_independent_scaling(ci):
    constant = max(abs(np.nanmin(ci)), abs(np.nanmax(ci)))
    scaled = (ci + constant) / (2 * constant)
    return scaled

def combine(im1, im2):
    return (im1 + im2) / 2

### pipelines


# Define a pipeline function that chains multiple image operations
def image_pipeline(image):
    grayscale_image = grayscale(image)
    resized_image = resize(grayscale_image, (100, 100))
    blurred_image = gaussian_blur(resized_image, sigma=1)
    return blurred_image

# JIT compile the pipeline function for performance
jit_image_pipeline = jit(image_pipeline)

# Now you can use the compiled function on your image data
processed_image = jit_image_pipeline(input_image)