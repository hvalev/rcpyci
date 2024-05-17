"""
File Name: pipelines.py
Description: TBD
"""
import numpy as np
from im_ops import apply_constant_scaling, apply_independent_scaling, apply_mask, combine, get_image_size
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_1samp

### pipelines for postprocessing classification images

default_ci_postprocessing_pipeline_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None
}

def ci_postprocessing_pipeline(
    base_image, 
    ci, 
    stimuli_params, 
    responses, 
    patches, 
    patch_idx, 
    mask=None, 
    scaling='independent', 
    scaling_constant=0.1):
    """
    Postprocess a classification image (ci) using various scaling and masking techniques.

    Parameters:
        base_image: The original image.
        ci: The classification image to postprocess.
        stimuli_params: TBD
        responses: TBD
        patches: TBD
        patch_idx: TBD
        mask: Optional mask to apply to the classification image. Defaults to None.
        scaling: One of 'independent' or 'constant'. If 'independent', each pixel in ci is scaled independently. If 'constant', all pixels are scaled by a constant factor. Defaults to 'independent'.
        scaling_constant: The constant scaling factor to use if scaling='constant'. Defaults to 0.1.

    Returns:
        combined: A postprocessed classification image.
    """
    if mask is not None:
        ci = apply_mask(ci, mask)
    scaled = apply_independent_scaling(ci) if scaling == 'independent' else \
             apply_constant_scaling(ci, scaling_constant)
    combined = combine(base_image, scaled)
    return combined

### pipelines for postprocessing classification images

default_compute_zmap_ci_pipeline_kwargs = {
    'sigma': 25,
    'threshold': 5,
}


def compute_zmap_ci_pipeline(base_image, ci, stimuli_params, responses, patches, patch_idx, sigma, threshold):
    """
    Compute a z-score map from a classification image (CI) by applying Gaussian filtering and thresholding.

    Parameters:
        base_image: The original image.
        ci: The classification image to compute the z-map for.
        stimuli_params: TBD
        responses: TBD
        patches: TBD
        patch_idx: TBD
        sigma: The standard deviation of the Gaussian filter. Defaults to 25.
        threshold: The threshold value above which values in the z-map are set to NaN. Defaults to 5.

    Returns:
        zmap: A z-score map computed from the input CI.
    """
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    zmap = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return zmap

default_compute_zmap_ttest_pipeline_kwargs = {}

def compute_zmap_ttest_pipeline(base_image, ci, stimuli_params, responses, patches, patch_idx):
    """
    Compute a z-score map from a classification image (CI) by applying t-test and thresholding.

    Parameters:
        base_image: The original image.
        ci: The classification image to compute the z-map for.
        stimuli_params: TBD
        responses: TBD
        patches: TBD
        patch_idx: TBD

    Returns:
        zmap: A z-score map computed from the input CI.
    """
    from core import __generate_individual_noise_stimulus
    img_size = get_image_size(base_image)
    weighted_parameters = stimuli_params * responses
    n_observations = len(responses)
    noise_images = np.zeros((img_size, img_size, n_observations))
    for obs in range(n_observations):
        noise_images[:, :, obs] = __generate_individual_noise_stimulus(weighted_parameters[obs], patches, patch_idx)
    t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    zmap = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
    return zmap
