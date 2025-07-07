import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, t, ttest_1samp

from .im_ops import apply_constant_scaling, apply_independent_scaling, apply_mask, combine, find_clusters
from .utils import cache_as_image, cache_as_numpy

### Existing Pipelines
compute_anti_ci_kwargs = {'anti_ci': True, 'use_cache': True, 'save_folder': 'anti_ci_raw'}
combine_anti_ci_kwargs = {'scaling': 'independent', 'scaling_constant': 0.1, 'mask': None, 'use_cache': True, 'save_folder': 'anti_ci'}

compute_ci_kwargs = {'anti_ci': False, 'use_cache': True, 'save_folder': 'ci_raw'}
combine_ci_kwargs = {'scaling': 'independent', 'scaling_constant': 0.1, 'mask': None, 'use_cache': True, 'save_folder': 'ci'}

compute_zscore_ci_kwargs = {'sigma': 5, 'use_cache': True, 'save_folder': 'zscore_ci'}
compute_ci_pixel_test_kwargs = {'FWHM': 4, 'p': 0.05, 'threshold': 2.7, 'save_folder': 'ci_rft_pixel_test'}
compute_ci_clusters_test_kwargs = {'use_cache': True, 'FWHM': 4, 'threshold': 2.7, 'p': 0.05, 'save_folder': 'ci_rft_cluster_test'}

compute_zscore_stim_params_kwargs = {'use_cache': True, 'save_folder': 'zscore_stim_params'}
compute_stim_params_clusters_test_kwargs = {'use_cache': True, 'FWHM': 4, 'threshold': 2.7, 'p': 0.05, 'save_folder': 'stim_params_rft_cluster_test'}

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


@cache_as_numpy
def compute_zscore_stimulus_params_memory_efficient(img_size, stimuli_params, responses, patches, patch_idx, cache_path=None):
    """
    Computes the Z-score image for stimulus parameters using running mean and variance.
    More memory efficient, but may incurr accumulating errors for medium datasets.
    """
    from .core import __generate_individual_noise_stimulus
    n_observations = len(responses)
    
    # Running mean and variance computation
    mean_accum = np.zeros((img_size, img_size))
    M2_accum = np.zeros((img_size, img_size))  # Sum of squared differences from the mean
    
    for obs in range(n_observations):
        noise_image = __generate_individual_noise_stimulus(stimuli_params[obs] * responses[obs], patches, patch_idx)
        # Welford's algorithm for running mean and variance
        delta = noise_image - mean_accum
        mean_accum += delta / (obs + 1)
        M2_accum += delta * (noise_image - mean_accum)  # Second pass for variance
    
    # Compute standard deviation from accumulated variance
    variance = M2_accum / (n_observations - 1)
    std_dev = np.sqrt(variance)
    
    # Compute t-statistics
    t_stat = mean_accum / (std_dev / np.sqrt(n_observations))
    
    # Compute p-values (two-tailed test)
    p_values = 2 * (1 - t.cdf(np.abs(t_stat), df=n_observations - 1))
    
    # Compute Z-score image (convert t-stat to Z-score)
    zscore_image = norm.ppf(1 - p_values / 2) * np.sign(t_stat)
    
    return {'zscore_image': zscore_image, 't_stat': t_stat, 'p_values': p_values}


@cache_as_numpy
def compute_rft_pixel_test(zscore_image, FWHM, p=0.05, cache_path=None):
    """
    Performs the pixel-level test based on Random Field Theory (RFT) to identify significant
    pixels in the Z-score image, while separating positive and negative significant pixels.
    
    Args:
        zscore_image (numpy.ndarray): Z-scored classification image.
        FWHM (float): Full width at half maximum of the Gaussian filter.
        p (float): Desired p-value threshold (default 0.05).
    
    Returns:
        dict: Contains the threshold used, and boolean arrays indicating significant pixels for both
              positive and negative Z-scores.
    """
    # Calculate resels (resolution elements) based on image volume and FWHM
    volume = zscore_image.size
    
    def EC0(t):
        return norm.sf(t)

    def EC1(t):
        # For 2D field with proper scale factors
        return (4 * np.log(2) / FWHM) * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)

    def EC2(t):
        # For 2D field with proper scale factors
        return (4 * np.log(2) / (FWHM**2)) * np.exp(-t**2 / 2) / (2 * np.pi)
    
    def p_max(t):
        # For a 2D image with proper weightings
        num_resels_2D = (volume / (FWHM ** 2))
        num_resels_1D = (4 * np.sqrt(volume)) / FWHM
        num_resels_0D = 1  # Typically 1 for a single connected field
        
        return num_resels_0D * EC0(t) + num_resels_1D * EC1(t) + num_resels_2D * EC2(t)
    
    # Find the threshold using binary search
    low, high = 0.0, 10.0
    tolerance = 1e-6
    for _ in range(512):
        mid = (low + high) / 2
        p_val = p_max(mid)
        if p_val < p:
            high = mid
        else:
            low = mid
        if high - low < tolerance:
            threshold = high
            break
    
    # Identify significant pixels for both positive and negative Z-scores
    significant_positive_pixels = zscore_image > threshold
    significant_negative_pixels = zscore_image < -threshold

    # Return the threshold and significant pixels for both positive and negative Z-scores
    return {
        'threshold': threshold,
        'significant_pixels_positive': significant_positive_pixels,
        'significant_pixels_negative': significant_negative_pixels
    }


@cache_as_numpy
def compute_rft_clusters_test(zscore_image, FWHM, threshold, p=0.05, cache_path=None):
    """
    Performs the cluster test based on RFT to find significant clusters in a Z-score image.
    
    Args:
        zscore_image (numpy.ndarray): Z-scored classification image.
        FWHM (float): Full width at half maximum of the Gaussian filter.
        threshold (float, optional): Cluster-forming threshold (e.g., 2.7).
        p (float, optional): Desired p-value threshold (e.g., 0.05).
    
    Returns:
        dict: Contains threshold, significant clusters, and positive/negative clusters.
    """
    # Compute Resels and EC densities (for 2D)
    volume = zscore_image.size  # Total number of pixels
    resels = volume / (FWHM ** 2)
    
    # EC densities for 2D Gaussian field (from Worsley et al. 1996)
    def EC2(t):
        # For 2D field with proper scale factors
        return (4 * np.log(2) / (FWHM**2)) * np.exp(-t**2 / 2) / (2 * np.pi)
    
    def p_cluster(k, t):
        # More accurate approximation for cluster-extent p-values including FWHM
        expected_clusters = resels * EC2(t)
        return np.exp(-expected_clusters) * np.exp(-k * (t**2 - 1) / (2 * (FWHM**2)))
    
    max_k = zscore_image.size

    # Find minimum cluster size using binary search
    low_k, high_k = 1, max_k
    while low_k < high_k:
        mid_k = (low_k + high_k) // 2
        if p_cluster(mid_k, threshold) < p:
            high_k = mid_k
        else:
            low_k = mid_k + 1
    min_cluster_size = low_k

    significant_clusters_positive = find_clusters(zscore_image > threshold, min_cluster_size)
    significant_clusters_negative = find_clusters(zscore_image < -threshold, min_cluster_size)

    return {'threshold': threshold, 
            'significant_clusters_positive': significant_clusters_positive, 
            'significant_clusters_negative': significant_clusters_negative}



pipeline_compute_ci_antici = [
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
]

pipeline_zmap_pixelwise_ci = [
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
    (compute_zscore_ci, compute_zscore_ci_kwargs),
    (compute_rft_pixel_test, compute_ci_pixel_test_kwargs)
]

pipeline_zmap_on_stim_param = [
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
    (compute_zscore_stimulus_params, compute_zscore_stim_params_kwargs),
    (compute_rft_clusters_test, compute_stim_params_clusters_test_kwargs)
]

full_pipeline = [
    # basic processing of computing the ci, antici
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
    # here we compute the ci zmap on the image (with gaussian blurring) as well as identify significant pixels or clusters on the ci image
    (compute_zscore_ci, compute_zscore_ci_kwargs),
    (compute_rft_pixel_test, compute_ci_pixel_test_kwargs),
    (compute_rft_clusters_test, compute_ci_clusters_test_kwargs),
    # here we compute the zmap on the parameter space (stimulus) and compute clusters based off of that zmap
    (compute_zscore_stimulus_params_memory_efficient, compute_zscore_stim_params_kwargs),
    (compute_rft_clusters_test, compute_stim_params_clusters_test_kwargs)
]