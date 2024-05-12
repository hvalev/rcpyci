"""
File Name: pipelines.py
Description: TBD
"""
import numpy as np
from im_ops import apply_constant_scaling, apply_independent_scaling, apply_mask, combine, get_image_size
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_1samp

# default values for generating stimuli
default_stimuli_generation_kwargs = {
    'n_trials': 5,
    'n_scales': 5,
    'sigma': 25,
    'noise_type': 'sinusoid',
}

### pipelines for postprocessing classification images

# fetch and overwrite params when needed
default_ci_postprocessing_pipeline_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None
}

def ci_postprocessing_pipeline(base_image, ci, stimuli_params, responses, patches, patch_idx, mask, scaling, scaling_constant):
    if mask is not None:
        ci = apply_mask(ci, mask)
    scaled = apply_independent_scaling(ci)
    if scaling == 'independent':
        scaled = apply_independent_scaling(ci)
    elif scaling == 'constant':
        scaled = apply_constant_scaling(ci, scaling_constant)
    combined = combine(base_image, scaled)
    return combined

### pipelines for postprocessing classification images

default_compute_zmap_ci_pipeline_kwargs = {
    'sigma': 25,
    'threshold': 5,
}

def compute_zmap_ci_pipeline(base_image, ci, stimuli_params, responses, patches, patch_idx, sigma, threshold):
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    zmap = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return zmap

default_compute_zmap_ttest_pipeline_kwargs = {}

def compute_zmap_ttest_pipeline(base_image, ci, stimuli_params, responses, patches, patch_idx):
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
