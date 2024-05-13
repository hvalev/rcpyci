"""
File Name: im_ops.py
Description: TBD
"""
import os

import numpy as np
from PIL import Image


### save/load image operations
# this function saves an unscaled [0,1] numpy array to an image
def save_image(image: np.array, path, clip=True, scale=True, save_npy=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if save_npy:
        npy_path = os.path.splitext(path)[0] + ".npy"
        np.save(npy_path, image)
    if clip:
        image = np.clip(image, 0, 1)
    if scale:
        image = image * 255
    Image.fromarray(image.astype(np.uint8)).save(path)

# rescaling image to utilize the full [0,255] range
def read_image(filename, grayscale=True, maximize_contrast=True):
    img = Image.open(filename)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    if maximize_contrast:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.asarray(img)

### image processing operators
def get_image_size(image):
    assert len(image.shape) == 2
    assert image.shape[0] == image.shape[1]
    return image.shape[0]

#TODO mask here needs to already be array, so it's jittable
#move read_image to the pipeline
def apply_mask(ci: np.ndarray, mask: np.ndarray):
    if isinstance(mask, str):
        mask_matrix = read_image(mask, grayscale=True)
    elif isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_matrix = mask
    else:
        raise ValueError("The mask argument is neither a path to file nor a 2D matrix!")
    masked_ci = np.ma.masked_where(mask_matrix == 0, ci)
    return masked_ci

def apply_constant_scaling(ci: np.ndarray, constant: np.ndarray):
    scaled = (ci + constant) / (2 * constant)
    if np.any((scaled > 1.0) | (scaled < 0)):
        print("Chosen constant value for constant scaling made noise "
              "of classification image exceed possible intensity range "
              "of pixels (<0 or >1). Choose a lower value, or clipping "
              "will occur.")
    return scaled

def apply_matched_scaling(ci: np.ndarray, base: np.ndarray):
    min_base = np.min(base)
    max_base = np.max(base)
    min_ci = np.min(ci[~np.isnan(ci)])
    max_ci = np.max(ci[~np.isnan(ci)])
    scaled = min_base + ((max_base - min_base) * (ci - min_ci) / (max_ci - min_ci))
    return scaled

def apply_independent_scaling(ci: np.ndarray):
    constant = max(abs(np.nanmin(ci)), abs(np.nanmax(ci)))
    scaled = (ci + constant) / (2 * constant)
    return scaled

def combine(im1: np.ndarray, im2: np.ndarray):
    return (im1 + im2) / 2
