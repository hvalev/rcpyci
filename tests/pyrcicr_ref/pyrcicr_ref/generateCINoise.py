from generateNoiseImage import generate_noise_image
# TODO: here I must make sure that the input variables are already numpy arrays...
def generate_ci_noise(stimuli, responses, p):
    # normalize responses, so we can multiply correctly
    responses = responses.reshape((responses.shape[0],1))
    # below original function
    weighted = stimuli * responses
    if weighted.ndim == 1:
        params = weighted
    else:
        params = weighted.mean(axis=0)
    
    return generate_noise_image(params, p)
