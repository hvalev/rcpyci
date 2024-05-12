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


# fetch and overwrite params when needed
default_ci_postprocessing_pipeline_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None
}

quick_zmap_kwargs = {}

ttest_zmap_kwargs = {}


### pipelines
def ci_postprocessing_pipeline(base_image, ci, mask, scaling, scaling_constant):
    if mask is not None:
        ci = apply_mask(ci, mask)
    scaled = apply_independent_scaling(ci)
    if scaling == 'independent':
        scaled = apply_independent_scaling(ci)
    elif scaling == 'constant':
        scaled = apply_constant_scaling(ci, scaling_constant)
    combined = combine(base_image, scaled)
    return combined

def compute_zmap_ci_pipeline():
    return 

def compute_zmap_parameter_pipeline():
    return
# JIT compile the pipeline function for performance
# jit_image_pipeline = jit(default_ci_pipeline)
# a = jit_image_pipeline

# Now you can use the compiled function on your image data
#processed_image = jit_image_pipeline(input_image)

