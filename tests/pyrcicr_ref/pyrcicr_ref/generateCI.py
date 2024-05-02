import os
import numpy as np
from scipy.stats import ttest_1samp, norm
from scipy.ndimage import gaussian_filter
from generate_noise import generate_sinusoid, generate_gabor, generate_noise_pattern, generate_scales
from generateNoiseImage import generate_noise_image
from generateCINoise import generate_ci_noise
from PIL import Image

# Main function
def generate_CI(stimuli, responses, baseimage, rdata, participants=None,
                save_individual_cis=False, save_as_png=True, filename='',
                targetpath='./cis', anti_CI=False, scaling='independent',
                scaling_constant=0.1, individual_scaling='independent',
                individual_scaling_constant=0.1, zmap=False,
                zmapmethod='quick', zmapdecoration=True, sigma=3,
                threshold=3, zmaptargetpath='./zmaps', n_cores=None,
                mask=None):
    
    # Preprocessing
    
    # Load parameter file (created when generating stimuli)
    npzfile = np.load(rdata)




    params = npzfile['patches']
    p = {}
    p['patches'] = npzfile['patches']
    p['patchIdx'] = npzfile['patchIdx']
    
    base_faces = {}
    stimuli_params = {}
    base_faces[baseimage] = npzfile['base_faces_'+baseimage]
    stimuli_params[baseimage] = npzfile['stimuli_params_'+baseimage]
    # Loop through the keys and build dictionaries
    # keys = list(npzfile.keys())
    # for key in keys:
    #     if key.startswith('base_faces_'):
    #         new_key = key.replace('base_faces_', '')
    #         base_faces[new_key] = npzfile[key]
    #     elif key.startswith('stimuli_params_'):
    #         new_key = key.replace('stimuli_params_', '')
    #         stimuli_params[new_key] = npzfile[key]

    #params = npzfile['s'] if 's' in npzfile else npzfile['p']['patches']
    # base_faces = npzfile['base_faces']
    # stimuli_params = npzfile['stimuli_params']
    # img_size = npzfile['img_size']
    
    # if participants is None:
    #     participants = np.arange(len(responses))
    #     aggregated_responses = np.array([np.mean(responses[stimuli == s]) for s in np.unique(stimuli)])
    #     responses = aggregated_responses
    #     stimuli = np.unique(stimuli)
    
    import pandas as pd

    # Check if all participants are NA
    if pd.isna(participants):
        # Create a DataFrame with responses and stimuli
        data = pd.DataFrame({'responses': responses, 'stimuli': stimuli})

        # Group by 'stimuli' and calculate the mean of 'responses'
        aggregated = data.groupby('stimuli')['responses'].mean().reset_index()

        # Update 'responses' and 'stimuli' with aggregated values
        responses = aggregated['responses'].to_numpy()#.tolist()
        stimuli = aggregated['stimuli'].to_numpy()#.tolist()


    base = base_faces[baseimage]
    if base is None:
        raise ValueError(f"File specified in rdata argument did not contain any reference to base image label: {baseimage}")
    

    params = stimuli_params[baseimage][stimuli]

    # params <- stimuli_params[[baseimage]][stimuli,]

    if params.shape[1] == 4096:
        params = params[:, :4092]
    
    # Generate CI(s)
    if participants is None:
        ci = generate_ci_noise(params, responses, p)
    else:
        # For each participant, construct the noise pattern
        pid_cis = []
        for pid in np.unique(participants):
            pid_rows = participants == pid
            ci = generate_ci_noise(params[pid_rows], responses[pid_rows], p)
            pid_cis.append(ci)
        ci = np.mean(pid_cis, axis=0)
    
    if mask is not None:
        ci = apply_mask(ci, mask)
    
    scaled = apply_scaling(base, ci, scaling, scaling_constant)
    combined = combine(scaled, base)
    
    if save_as_png:
        save_to_image(baseimage, combined, targetpath, filename, anti_CI)
    
    # Z-map
    zmap_image = None
    if zmap:
        if zmapmethod == 'quick':
            zmap_image = process_quick_zmap(ci, sigma, threshold)
        elif zmapmethod == 't.test':
            assert ci.shape[1] == ci.shape[2]
            img_size = ci.shape[1]
            zmap_image = process_ttest_zmap(params, responses, p, img_size, ci, n_cores=1, pid_cis=None)
        else:
            raise ValueError(f"Invalid zmap method: {zmapmethod}")
        #return zmap
        #zmap_image = create_zmap(ci, zmapmethod, sigma, threshold)
        #plot_zmap(zmap, combined, baseimage, sigma, threshold, zmapdecoration, zmaptargetpath)
    
    return {'ci': ci, 'scaled': scaled, 'base': base, 'combined': combined, 'zmap': zmap_image}


# Functions
import numpy as np
from scipy.stats import norm, ttest_1samp
from scipy.ndimage import gaussian_filter

def process_quick_zmap(ci, sigma, threshold):
    # Blur CI
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    # Create z-map
    scaled_image = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    # Apply threshold
    zmap = scaled_image.copy()
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return {"blurred_ci": blurred_ci, "scaled_image": scaled_image, "zmap": zmap}

def process_ttest_zmap(params, responses,img_size, ci, rdata, n_cores=1, pid_cis=None):
    # Load parameter file (created when generating stimuli)
    npzfile = np.load(rdata)
    #params = npzfile['patches']
    p = {}
    p['patches'] = npzfile['patches']
    p['patchIdx'] = npzfile['patchIdx']
    if pid_cis is None:
        responses = responses.reshape((responses.shape[0],1))
        weighted_parameters = params * responses
        n_observations = len(responses)
        noise_images = np.zeros((img_size, img_size, n_observations))
        # For each observation, generate noise image
        for obs in range(n_observations):
            noise_image = generate_noise_image(weighted_parameters[obs], p)
            noise_images[:, :, obs] = noise_image
    else:
        noise_images = pid_cis
    # Compute p-value for each pixel using t-test
    reshaped_images = noise_images.reshape(-1, noise_images.shape[-1])

    # Apply t-test to each pixel (element) of the reshaped 2D array
    # and extract p-values
    # below is a super slow version, but linguistically similar to the R implementation
    # p_values = np.apply_along_axis(lambda x: ttest_1samp(x, popmean=0)[1], axis=1, arr=reshaped_images)
    # below is a much faster identical version
    t_stat, p_values = ttest_1samp(reshaped_images, popmean=0, axis=1)
    # Reshape p_values to the original shape
    pmap = p_values.reshape(noise_images.shape[:-1])
    # Create Z-map
    zmap = np.sign(ci) * np.abs(norm.ppf(pmap / 2))
    # nai burzata bez reshape
    #t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    #pmap = p_values
    #zmap = np.sign(ci) * np.abs(norm.ppf(pmap / 2))
    return zmap

def apply_mask(ci, mask):
    # mask is a path to a file
    if isinstance(mask, str):
        mask_matrix = np.array(Image.open(mask).convert('L'))
    elif isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_matrix = mask
    else:
        raise ValueError("The mask argument is neither a string nor a 2D matrix!")

    if mask_matrix.shape != ci.shape:
        raise ValueError(f"Mask is not of the same dimensions as the CI: {ci.shape} vs {mask_matrix.shape}")

    if np.any((mask_matrix != 0) & (mask_matrix != 255)):
        raise ValueError("The mask contains values other than 0 (black) or 255 (white)!")
    
    masked_ci = np.where(mask_matrix == 0, np.nan, ci)
    return masked_ci


def apply_scaling(base, ci, scaling, constant):
    if scaling == 'none':
        scaled = ci
    elif scaling == 'constant':
        scaled = (ci + constant) / (2 * constant)
        if np.any((scaled > 1.0) | (scaled < 0)):
            print("Chosen constant value for constant scaling made noise "
                  "of classification image exceed possible intensity range "
                  "of pixels (<0 or >1). Choose a lower value, or clipping "
                  "will occur.")
    elif scaling == 'matched':
        min_base = np.min(base)
        max_base = np.max(base)
        min_ci = np.min(ci[~np.isnan(ci)])
        max_ci = np.max(ci[~np.isnan(ci)])
        scaled = min_base + ((max_base - min_base) * (ci - min_ci) / (max_ci - min_ci))
    elif scaling == 'independent':
        constant = max(abs(np.nanmin(ci)), abs(np.nanmax(ci)))
        scaled = (ci + constant) / (2 * constant)
    else:
        print(f"Scaling method '{scaling}' not found. Using none.")
        scaled = ci

    return scaled

def combine(scaled, base):
    return (scaled + base) / 2

def save_to_image(baseimage, combined, targetpath, filename, anti_CI):
    filename = filename or baseimage
    if anti_CI:
        filename = 'python_antici_' + filename
    else:
        filename = 'python_ci_' + filename
    filename += '.png'
    os.makedirs(targetpath, exist_ok=True)
    Image.fromarray((combined * 255).astype(np.uint8)).save(os.path.join(targetpath, filename))

# def create_zmap(ci, zmapmethod, sigma, threshold):
#     if zmapmethod == 'quick':
#         zmap = gaussian_filter(ci, sigma=sigma)
#         zmap = (zmap - np.mean(zmap)) / np.std(zmap)
#         zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
#     elif zmapmethod == 't.test':
#         t_stat, _ = ttest_1samp(ci.flatten(), 0)
#         p_values = norm.sf(np.abs(t_stat)) * 2
#         zmap = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
#     else:
#         raise ValueError(f"Invalid zmap method: {zmapmethod}")
#     return zmap

# def plot_zmap(zmap, bgimage, filename, sigma, threshold, decoration, zmaptargetpath):
#     pass  # Implementation needed
