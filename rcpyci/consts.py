### file for containing standard values for certain functions
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