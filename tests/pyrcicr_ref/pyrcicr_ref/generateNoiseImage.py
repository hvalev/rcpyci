import numpy as np


def generate_noise_image(params, p):
    if len(params) != np.max(p['patchIdx']):
        if len(params) == (np.max(p['patchIdx']) + 1) and np.min(p['patchIdx']) == 0:
            # Some versions of dependencies created patch indices starting with 0, while others start at 1. Adjust for this mismatch.
            print("Warning: Python data patch indices start at 0, whereas parameters are used from position 1. Due to this mismatch, one sinusoid will not be shown in the resulting noise image.")
        else:
            raise ValueError("Stimulus generation aborted: the number of parameters doesn't equal the number of patches!")

    if 'sinusoids' in p:
        # Pre-0.3.3 noise pattern, rename for appropriate use
        p['patches'] = p['sinusoids']
        p['patchIdx'] = p['sinIdx']
        p['noise_type'] = 'sinusoid'

    # Create the noise image
    #noise = np.mean(p['patches'] * params[p['patchIdx']], axis=2)


    #noise <- apply(p$patches * array(params[p$patchIdx], dim(p$patches)), 1:2, mean)
    # another way
    #patch_indices = p['patchIdx']
    # we need to convert to int and subtract 1 to make it 0-indexed
    patch_indices = p['patchIdx'].astype(int) - 1
    pd = p['patches'].shape
    patch_params = np.array(params[patch_indices]).reshape(pd)
    reshaped_matrix = (p['patches'] * patch_params).reshape((pd[0]*pd[1], pd[2]))
    improved_noise = np.mean(reshaped_matrix, axis=1).reshape(pd[0:2])

    return improved_noise

# Example usage
# params = np.random.randn(4092)
# p = generate_noise_pattern(img_size=256)
# noise_image = generate_noise_image(params, p)
