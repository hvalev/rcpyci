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
    """
    Saves an unscaled [0,1] numpy array to an image.
    
    Parameters:
    - image (np.array): The input numpy array. Should be of shape (height, width).
    - path (str): The path where the image will be saved.
    - clip (bool): Whether or not to clip the pixel values between 0 and 1.
    - scale (bool): Whether or not to scale the pixel values from [0,1] to [0,255].
    - save_npy (bool): Whether or not to save the unscaled numpy array as a .npy file.

    Returns:
    None
    """
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
    """
    Reads an image file and preprocesses it.

    Parameters:
    - filename (str): The path to the image file.
    - grayscale (bool): Whether or not to convert the image to grayscale.
    - maximize_contrast (bool): Whether or not to rescale the image to utilize the full [0,255] range.

    Returns:
    np.array: The preprocessed image as a numpy array.
    """
    img = Image.open(filename)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    if maximize_contrast:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.asarray(img)

### image processing operators
def get_image_size(image) -> int:
    """
    Returns the size of an image and confirms it's a square image.

    This function returns the size (i.e., height and width) of a given image.

    Parameters:
    - image: The input image.

    Returns:
    int: A int containing the height and width of the image.
    """
    assert len(image.shape) == 2
    assert image.shape[0] == image.shape[1]
    return image.shape[0]

#TODO tests if adding a mask affects the computation
def apply_mask(ci: np.ndarray, mask: np.ndarray):
    """
    Applies a binary mask to an image.

    This function applies a given binary mask to the input image ci. Pixels in the mask with value 0 are set to zero in the resulting masked image.

    Parameters:
     - ci (np.ndarray): The input image.
     - mask (np.ndarray): A binary mask of the same shape as the input image.

    Returns:
     np.ndarray: The masked image.
    """
    if isinstance(mask, str):
        mask_matrix = read_image(mask, grayscale=True)
    elif isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_matrix = mask
    else:
        raise ValueError("The mask argument is neither a path to file nor a 2D matrix!")
    masked_ci = np.ma.masked_where(mask_matrix == 0, ci)
    return masked_ci

#TODO logging
def apply_constant_scaling(ci: np.ndarray, constant: np.ndarray): 
    """Applies a constant scaling to an image.

    This function applies a given constant value to the input image ci, effectively shifting its intensity values. 
    The new minimum and maximum pixel values are determined by the applied constant, ensuring that all pixel values remain within the 0-1 range.

    Parameters:
     - ci (np.ndarray): The input image.
     - constant (np.ndarray): A scalar value used for scaling the input image.

    Returns:
    np.ndarray: The scaled image.
    """
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

def find_clusters(mask, min_size:int=1):
    """
    Identifies contiguous clusters in a binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask where True indicates regions above threshold.
        min_size (int, optional): Minimum size of clusters to retain.
    
    Returns:
        numpy.ndarray: Boolean mask of significant clusters.
    """
    import scipy.ndimage as ndi
    labeled, _ = ndi.label(mask)
    sizes = np.bincount(labeled.ravel())[1:]
    mask_sizes = np.where(sizes >= min_size)[0] + 1
    significant_clusters = np.isin(labeled, mask_sizes)
    return significant_clusters