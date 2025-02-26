import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_1samp, t

from .im_ops import apply_constant_scaling, apply_independent_scaling, apply_mask, combine, find_clusters
from .utils import cache_as_image, cache_as_numpy

### Existing Pipelines
compute_anti_ci_kwargs = {'anti_ci': True, 'use_cache': True, 'save_folder': 'anti_ci_raw'}
combine_anti_ci_kwargs = {'scaling': 'independent', 'scaling_constant': 0.1, 'mask': None, 'use_cache': True, 'save_folder': 'anti_ci'}

compute_ci_kwargs = {'anti_ci': False, 'use_cache': True, 'save_folder': 'ci_raw'}
combine_ci_kwargs = {'scaling': 'independent', 'scaling_constant': 0.1, 'mask': None, 'use_cache': True, 'save_folder': 'ci'}

compute_zscore_ci_kwargs = {'sigma': 5, 'use_cache': True, 'save_folder': 'zscore_ci'}
compute_zscore_stim_params_kwargs = {'use_cache': True, 'save_folder': 'zscore_stim_params'}

@cache_as_numpy
def compute_ci(base_image, stimuli_params, responses, patches, patch_idx, anti_ci, n_trials, n_scales, gabor_sigma, noise_type, seed, cache_path=None):
    from .core import compute_ci
    return {'ci': compute_ci(base_image, responses, stimuli_params, patches, patch_idx, anti_ci, n_trials, n_scales, gabor_sigma, noise_type, seed)}

@cache_as_image
def combine_ci(base_image, ci, mask=None, scaling='independent', scaling_constant=0.1, cache_path=None):
    if mask is not None:
        ci = apply_mask(ci, mask)
    scaled = apply_independent_scaling(ci) if scaling == 'independent' else apply_constant_scaling(ci, scaling_constant)
    return {'combined': combine(base_image, scaled)}

@cache_as_numpy
def compute_zscore_ci(ci, sigma=None, cache_path=None):
    """
    Computes the Z-score image for the CI (classification image).
    """
    # Blurring
    if sigma is not None and isinstance(sigma, (float, int)):
        ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    zscore_image = (ci - np.mean(ci)) / np.std(ci)
    return {'zscore_image': zscore_image}


@cache_as_numpy
def compute_zscore_stimulus_params(img_size, ci, stimuli_params, responses, patches, patch_idx, cache_path=None):
    """
    Computes the Z-score image for stimulus parameters based on t-tests.
    """
    from .core import __generate_individual_noise_stimulus
    weighted_parameters = stimuli_params * responses
    n_observations = len(responses)
    noise_images = np.zeros((img_size, img_size, n_observations))
    
    # Generate noise images for each observation
    for obs in range(n_observations):
        noise_images[:, :, obs] = __generate_individual_noise_stimulus(weighted_parameters[obs], patches, patch_idx)
    
    # Compute t-statistics and p-values
    t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    
    # Convert p-values to Z-scores
    zscore_image = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
    
    return {'zscore_image': zscore_image, 't_stat': t_stat, 'p_values': p_values}


### Compute infoval on a ci as a postprocessing pipeline

compute_infoval_2ifc_pipeline_kwargs = {
    'path_to_reference_norms': None,
    'use_cache': True,
    'save_folder': 'infoval'
}

@cache_as_numpy
def compute_infoval_2ifc_pipeline(ci, path_to_reference_norms, cache_path=None):
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
    (compute_zscore_ci, compute_zscore_ci_kwargs),
]