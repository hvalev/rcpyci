"""
File Name: pipelines.py
Description: TBD
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_1samp

from .im_ops import apply_constant_scaling, apply_independent_scaling, apply_mask, combine, get_image_size
from .utils import cache_as_image, cache_as_numpy

### pipelines for postprocessing classification images
compute_anti_ci_kwargs = {
    'anti_ci': True,
    'use_cache': True,
    'save_folder': 'anti_ci_raw'
}

compute_ci_kwargs = {
    'anti_ci': False,
    'use_cache': True,
    'save_folder': 'ci_raw'
}

@cache_as_numpy
def compute_ci(base_image, stimuli_params, responses, patches, patch_idx, anti_ci, n_trials, n_scales, gabor_sigma, noise_type, seed, cache=None):
    from .core import compute_ci
    ci = compute_ci(base_image=base_image,
                    responses=responses,
                    stimuli_params=stimuli_params,
                    patches=patches,
                    patch_idx=patch_idx,
                    anti_ci=anti_ci,
                    n_trials=n_trials,
                    n_scales=n_scales,
                    gabor_sigma=gabor_sigma,
                    noise_type=noise_type,
                    seed=seed)
    return {'ci': ci}


combine_anti_ci_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None,
    'use_cache': True,
    'save_folder': 'anti_ci'
}

combine_ci_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None,
    'use_cache': True,
    'save_folder': 'ci'
}

@cache_as_image
def combine_ci(base_image, ci, mask=None, scaling='independent', scaling_constant=0.1, cache=None):
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
    return {'combined': combined}

### pipelines for postprocessing classification images

compute_zmap_ci_kwargs = {
    'threshold': 5,
    'sigma': 5,
    'use_cache': True,
    'save_folder': 'zmap_ci'
}

@cache_as_numpy
def compute_zmap_ci(ci, sigma, threshold, cache=None):
    """
    Compute a z-score map from a classification image (CI) by applying Gaussian filtering and thresholding.

    Parameters:
        base_image: The original image.
        ci: The classification image to compute the z-map for.
        sigma: The standard deviation of the Gaussian filter. Defaults to 25.
        threshold: The threshold value above which values in the z-map are set to NaN. Defaults to 5.

    Returns:
        zmap: A z-score map computed from the input CI.
    """
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    zmap = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return {'zmap': zmap}

compute_zmap_stimulus_params_kwargs = {
    'use_cache': True,
    'save_folder': 'zmap_stim'
}

@cache_as_numpy
def compute_zmap_stimulus_params(base_image, ci, stimuli_params, responses, patches, patch_idx, cache=None):
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
    from .core import __generate_individual_noise_stimulus
    img_size = get_image_size(base_image)
    weighted_parameters = stimuli_params * responses
    n_observations = len(responses)
    noise_images = np.zeros((img_size, img_size, n_observations))
    for obs in range(n_observations):
        noise_images[:, :, obs] = __generate_individual_noise_stimulus(weighted_parameters[obs], patches, patch_idx)
    t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    zmap = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
    return {'zmap' : zmap, 't_stat': t_stat, 'p_values': p_values}


### Compute infoval on a ci as a postprocessing pipeline

compute_infoval_2ifc_pipeline_kwargs = {
    'path_to_reference_norms': None,
    'use_cache': True,
    'save_folder': 'infoval'
}

@cache_as_numpy
def compute_infoval_2ifc_pipeline(ci, path_to_reference_norms, cache=None):
    ref_norms = np.load(path_to_reference_norms)
    from .infoval import compute_info_val_2ifc
    info_val, cinorm, ref_median, ref_mad, ref_iter = compute_info_val_2ifc(target_ci=ci,
                                                                            reference_norms=ref_norms)
    
    return {
        "info_val": info_val,
        "cinorm": cinorm,
        "ref_median": ref_median,
        "ref_mad": ref_mad,
        "ref_iter": ref_iter
    }




full_pipeline = [
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
    (compute_zmap_ci, compute_zmap_ci_kwargs),
    (compute_zmap_stimulus_params, compute_zmap_stimulus_params_kwargs)
]