### file for containing standard values for certain functions
baseline_params_kwargs = {
    'n_trials': 5,
    'n_scales': 5,
}

# fetch and overwrite params when needed
default_pipeline_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None
}

quick_zmap_kwargs = {}

ttest_zmap_kwargs = {}