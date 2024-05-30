import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from .core import __generate_ci_noise, __generate_noise_pattern, __generate_stimuli_params
from .im_ops import get_image_size


def generate_reference_distribution_2ifc(base_image: np.ndarray, 
                                         n_trials: int = 770,
                                         n_scales: int = 5,
                                         gabor_sigma: int = 25,
                                         noise_type: str = 'sinusoid',
                                         seed: int = 1,
                                         save_path: str = None,
                                         stimuli_params: np.ndarray = None, 
                                         patches: np.ndarray = None,
                                         patch_idx: np.ndarray = None,
                                         iter: int = 10000):
    """
    Generate reference responses selection distribution for 2 ifc task with your stimuli params.

    This function simulates random responding in a 2-IFC task with n_trials for n iterations. Default is 10000.
    It generates the stimuli parameters, noise pattern, and computes the classification image (ci) norm for 
    each simulated response selection. The generated norms are returned as the reference distribution.
    You can regenerate the stimuli parameters by providing the seed and additional parameters used to generate 
    the stimulus images or provide the stimuli parameter space using the stimuli_params variable.

    Parameters:
        base_image: np.ndarray
            The base image used to generate the stimuli.
        n_trials: int
            Number of trials in the 2-IFC task. Defaults to 770.
        n_scales: int
            Number of scales for the Gabor noise pattern. Defaults to 5.
        gabor_sigma: int
            Sigma value for the Gabor noise pattern. Defaults to 25.
        noise_type: str
            Type of noise pattern to use ('sinusoid' or 'gabor'). Defaults to 'sinusoid'.
        seed: int
            Random seed used for generating stimuli and responses. Defaults to 1.
        stimuli_params: np.ndarray
            Optional parameter that can be provided directly instead of using the default value.
        patches: np.ndarray
            Optional patch pattern array used for generating noise. If not provided, it will be generated internally.
        patch_idx: np.ndarray
            Optional index array for the patch pattern. If not provided, it will be generated internally.
        iter: int
            Number of iterations to run the simulation. Defaults to 10000.

    Returns:
        reference_norms: list of float
            The computed norms of the classification images (ci) for each iteration, which form the reference distribution.
    """

    # Simulate random responding in 2IFC task with ntrials trials across iter iterations
    img_size = get_image_size(base_image)
    if iter < 10000:
        print("You should set iter >= 10000 for InfoVal statistic to be reliable")
    
    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials = n_trials, n_scales = n_scales, seed = seed)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size = img_size, noise_type = noise_type, n_scales = n_scales, gabor_sigma = gabor_sigma)

    reference_norms = []
    for _ in tqdm(range(iter), desc="Computing reference distribution"):
        # Generate random responses for this iteration
        responses = np.random.choice([-1, 1], size=n_trials).reshape(n_trials, 1)
        ci = __generate_ci_noise(stimuli_params, responses, patches, patch_idx)
        # Compute classication image for this iteration
        #ci = np.dot(stimuli, responses) / stimuli.shape[1]
        # Save norm for this iteration
        reference_norms.append(norm(ci, 'f'))
    results = np.array(reference_norms)
    if save_path:
        np.save(save_path, results)
    return results

def compute_info_val_2ifc(target_ci: np.ndarray, reference_norms):
    # Compute reference values
    ref_median = np.median(reference_norms)
    ref_mad = np.median(np.abs(reference_norms - ref_median))
    ref_iter = len(reference_norms)

    # Compute informational value metric
    cinorm = norm(target_ci, 'f')
    info_val = (cinorm - ref_median) / ref_mad

    # print(f"Informational value: z = {info_val} (ci norm = {cinorm}; reference median = {ref_median}; MAD = {ref_mad}; iterations = {ref_iter})")
    
    return info_val, cinorm, ref_median, ref_mad, ref_iter

