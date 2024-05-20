import numpy as np
from core import __generate_ci_noise, __generate_noise_pattern, __generate_stimuli_params
from numpy.linalg import norm
from tqdm import tqdm


# generate reference distribution for 2-IFC model. The stimuli params can be generated or sideloaded (sideloaded in the pipeline)
def generate_reference_distribution_2IFC(n_trials,
                                         img_size, 
                                         n_scales, 
                                         noise_type, 
                                         sigma, 
                                         seed=1, 
                                         stimuli_params = None, 
                                         patches = None,
                                         patch_idx = None,
                                         iter=10000):
    # Simulate random responding in 2IFC task with ntrials trials across iter iterations
    if iter < 10000:
        print("You should set iter >= 10000 for InfoVal statistic to be reliable")
    
    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials = n_trials, n_scales = n_scales, seed = seed)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size = img_size, noise_type = noise_type, n_scales = n_scales, sigma = sigma)

    reference_norms = []
    for _ in tqdm(range(iter), desc="Computing reference distribution"):
        # Generate random responses for this iteration
        responses = np.random.choice([-1, 1], size=n_trials).reshape(n_trials, 1)
        ci = __generate_ci_noise(stimuli_params, responses, patches, patch_idx)
        # Compute classication image for this iteration
        #ci = np.dot(stimuli, responses) / stimuli.shape[1]
        # Save norm for this iteration
        reference_norms.append(norm(ci, 'f'))
    return reference_norms

def compute_info_val_2IFC(target_ci, reference_norms, iter=10000):
    # Compute reference values
    ref_median = np.median(reference_norms)
    ref_mad = np.median(np.abs(reference_norms - ref_median))
    ref_iter = len(reference_norms)

    # Compute informational value metric
    cinorm = norm(target_ci, 'f')
    info_val = (cinorm - ref_median) / ref_mad

    print(f"Informational value: z = {info_val} (ci norm = {cinorm}; reference median = {ref_median}; MAD = {ref_mad}; iterations = {ref_iter})")
    
    return info_val, cinorm, ref_median, ref_mad, ref_iter

