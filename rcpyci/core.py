import math
import numpy as np
from scipy.stats import norm, ttest_1samp
from scipy.ndimage import gaussian_filter
import os
from PIL import Image
import random
from tqdm import tqdm
from datetime import datetime

#TODO make sure numpy functions are jaxified
#TODO static typing
#TODO split scaling/noise to be separate functions and define a way to propagate that into the information flow
#TODO split writing to files / caching to a separate function (write to numpy, write to image, write to whatever)

def generate_sinusoid(img_size: int, cycles: int, angle: int, phase: int, contrast: float):
    angle = math.radians(angle)
    x = np.linspace(0, cycles, img_size)
    y = np.linspace(0, cycles, img_size)
    X, Y = np.meshgrid(x, y)
    sinepatch = X * math.cos(angle) + Y * math.sin(angle)
    sinusoid = (sinepatch * 2 * math.pi) + phase
    sinusoid = contrast * np.sin(sinusoid)
    return sinusoid

def generate_gabor(img_size, cycles, angle, phase, sigma, contrast):
    sinusoid = generate_sinusoid(img_size, cycles, angle, phase, contrast)
    x0 = np.linspace(-0.5, 0.5, img_size)
    X, Y = np.meshgrid(x0, x0)
    gauss_mask = np.exp(-((X ** 2 + Y ** 2) / (2 * (sigma / img_size) ** 2)))
    gabor = gauss_mask * sinusoid
    return gabor

def generate_scales(img_size=512, nscales=5):
    scales = 2 ** np.arange(nscales)
    x, y = np.meshgrid(np.arange(1, img_size+1), np.arange(1, img_size+1))
    patch_size = x / y
    #TODO scales will be integers, so make sure to check that all are integer convertible
    #and then actually convert it to integers.
    return patch_size

def generate_noise_pattern(img_size=512, nscales=5, noise_type='sinusoid', sigma=25, pre_0_3_0=False):
    # Settings of sinusoids
    orientations = np.array([0, 30, 60, 90, 120, 150])
    phases = np.array([0, np.pi/2])
    scales = 2 ** np.arange(nscales)

    # Size of patches per scale
    #patch_size = np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1), np.arange(1, len(scales) + 1))[0] / np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1), np.arange(1, len(scales) + 1))[1]
    patch_size = generate_scales(img_size=img_size, nscales=nscales)
    
    # Number of patch layers needed
    nr_patches = len(scales) * len(orientations) * len(phases)

    # Preallocate memory
    patches = np.zeros((img_size, img_size, nr_patches))
    patch_idx = np.zeros((img_size, img_size, nr_patches))

    # Counters
    if pre_0_3_0:
        co = 0  # patch layer counter
        idx = 0  # contrast index counter
    else:
        co = 1  # patch layer counter
        idx = 1  # contrast index counter

    for scale in scales:
        for orientation in orientations:
            for phase in phases:
                # Generate single patch
                #size = patch_size[int(scale) - 1, :, :]
                size = patch_size[int(scale) - 1, img_size - 1]
                if noise_type == 'gabor':
                    p = generate_gabor(int(size), 1.5, int(orientation), phase, sigma, 1)
                else:
                    # img_size, cycles, angle, phase, contrast
                    p = generate_sinusoid(int(size), int(2), int(orientation), phase, 1)

                # Repeat to fill scale
                patches[:, :, co - 1] = np.tile(p, (scale, scale))

                # Create index matrix
                for col in range(1, scale + 1):
                    for row in range(1, scale + 1):
                        # Insert absolute index for later contrast weighting
                        patch_idx[int(size * (row - 1)):int(size * row), int(size * (col - 1)):int(size * col), co - 1] = idx

                        # Update contrast counter
                        idx += 1

                # Update layer counter
                co += 1

    return {'patches': patches, 'patchIdx': patch_idx, 'noise_type': noise_type, 'generator_version': '0.3.0'}



# Main function
def generate_CI(stimuli, 
                responses, 
                baseimage, 
                rdata, 
                participants=None,
                save_individual_cis=False, 
                save_as_png=True, 
                filename='',
                targetpath='./cis', 
                anti_CI=False, 
                scaling='independent',
                scaling_constant=0.1, 
                individual_scaling='independent',
                individual_scaling_constant=0.1, 
                zmap=False,
                zmapmethod='quick', 
                zmapdecoration=True, 
                sigma=3,
                threshold=3, 
                zmaptargetpath='./zmaps', 
                n_cores=None,
                mask=None):
    
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
        responses = aggregated['responses'].to_numpy()
        stimuli = aggregated['stimuli'].to_numpy()


    base = base_faces[baseimage]
    if base is None:
        raise ValueError(f"File specified in rdata argument did not contain any reference to base image label: {baseimage}")

    params = stimuli_params[baseimage][stimuli]

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
    
    return {'ci': ci, 'scaled': scaled, 'base': base, 'combined': combined, 'zmap': zmap_image}


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
    p_values = np.apply_along_axis(lambda x: ttest_1samp(x, popmean=0)[1], axis=1, arr=reshaped_images)
    # below is a much faster version of it
    # t_stat, p_values = ttest_1samp(reshaped_images, popmean=0, axis=1)
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


def generate_noise_image(params, p):
    if len(params) != np.max(p['patchIdx']):
        if len(params) == (np.max(p['patchIdx']) + 1) and np.min(p['patchIdx']) == 0:
            # Some versions of dependencies created patch indices starting with 0, while others start at 1. Adjust for this mismatch.
            print("Warning: Python data patch indices start at 0, whereas parameters are used from position 1. Due to this mismatch, one sinusoid will not be shown in the resulting noise image.")
        else:
            raise ValueError("Stimulus generation aborted: the number of parameters doesn't equal the number of patches!")
        
    #TODO make sure patchindices are already ints, so we don't need this
    # we need to convert to int and subtract 1 to make it 0-indexed
    patch_indices = p['patchIdx'].astype(int) - 1
    pd = p['patches'].shape
    patch_params = np.array(params[patch_indices]).reshape(pd)
    reshaped_matrix = (p['patches'] * patch_params).reshape((pd[0]*pd[1], pd[2]))
    improved_noise = np.mean(reshaped_matrix, axis=1).reshape(pd[0:2])
    return improved_noise


def read_image(filename, 
               grayscale=False):
    img = Image.open(filename)
    # Convert to grayscale
    if grayscale:
        img = img.convert('L')  
    return np.asarray(img)

def generate_stimuli_2IFC(base_face_files, 
                          overwrite_params=False, 
                          params_values=None, 
                          n_trials=770, 
                          img_size=512, 
                          stimulus_path='./stimuli', 
                          label='rcic', 
                          use_same_parameters=True, 
                          seed=1, 
                          maximize_baseimage_contrast=True, 
                          noise_type='sinusoid', 
                          nscales=5, 
                          sigma=25, 
                          ncores=None,
                          return_as_dataframe=False, 
                          save_as_png=True, 
                          save_rdata=True
                          ):
    # Initialize
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(stimulus_path, exist_ok=True)
    
    stimuli_params = {}
    base_faces = {}
    
    if not isinstance(base_face_files, dict):
        print("Please provide base face file names as a dictionary, e.g. base_face_files={'aName': 'baseface.jpg'}")
        return
    
    for base_face, filename in base_face_files.items():
        # Read base face
        #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE if noise_type == 'sinusoid' else cv2.IMREAD_COLOR)
        img = read_image(filename, grayscale=(noise_type == 'sinusoid'))
        
        # Check if base face is square
        if img.shape[0] != img.shape[1]:
            raise ValueError(f"Base face is not square! It's {img.shape[0]} by {img.shape[1]} pixels. Please use a square base face.")
        
        # Adjust size of base face
        # img = cv2.resize(img, (img_size, img_size))
        
        # If necessary, rescale to maximize contrast
        if maximize_baseimage_contrast:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            #img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Save base image to dictionary
        base_faces[base_face] = img
    
    # Compute number of parameters needed
    #nparams = sum(6*2*(2**(np.arange(nscales))))**2
    nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
    
    # Generate parameters
    if use_same_parameters:
        # if it exists and is not False use the passed params
        if overwrite_params:
            params = params_values
        else:
            # Generate stimuli parameters, one set for all base faces
            params = np.random.uniform(-1, 1, size=(n_trials, nparams))
        
        # Assign to each base face the same set
        # this is not great since later we reuse params variable
        # and it gets confusing that we are simply copying it over here
        for base_face in base_faces:
            stimuli_params[base_face] = params.copy()
    else:
        for base_face in base_faces:
            # Generate stimuli parameters, unique to each base face
            stimuli_params[base_face] = np.random.uniform(-1, 1, size=(n_trials, nparams))
    
    # Generate stimuli
    stimuli = np.zeros((n_trials, img_size, img_size))
    
    # TODO here in generate_noise_pattern I need to remove the indexing starting from 1
    # TODO so that in generate_noise_image, I can also do 0-indexing
    # Generate noise pattern
    p = generate_noise_pattern(img_size=img_size, 
                               noise_type=noise_type, 
                               nscales=nscales, 
                               sigma=sigma)

    #TODO jax tqdm?
    for trial in tqdm(range(n_trials)):
        for base_face in base_faces:
            params = stimuli_params[base_face][trial]
            
            
            #p = generate_noise_pattern(img_size, noise_type, nscales, sigma)
            noise_pattern = generate_noise_image(params, p)

            stimuli[trial,:,:] = noise_pattern
            
            # Scale noise
            stimulus = ((noise_pattern + 0.3) / 0.6)
            
            # Add base face
            combined = (stimulus + base_faces[base_face]) / 2.0
            
            # add +1 to trial name only so that we can be on par with
            # R's 1 indexing
            trial_adj = trial+1

            # Write to file
            if save_as_png:
                # TODO make sure to print when clipped
                # Clip values to range [0, 1]
                combined_clipped = np.clip(combined, 0, 1)
                # Scale the values to the range [0, 255] for PIL
                scaled_values = (combined_clipped * 255).astype(np.uint8)
                # Convert the numpy array to a PIL Image
                img = Image.fromarray(scaled_values, mode='L')
                # Define the filename based on your variables
                filename_ori = f"{label}_{base_face}_{seed:01d}_{trial_adj:05d}_p_ori.png"
                # Save the image as a PNG file
                img.save(os.path.join(stimulus_path, filename_ori))

            # Compute inverted stimulus
            stimulus = ((-noise_pattern + 0.3) / 0.6)
            
            # Add base face
            combined = (stimulus + base_faces[base_face]) / 2.0
            
            # Write to file
            if save_as_png:
                # TODO make sure to print when clipped
                # Clip values to range [0, 1]
                combined_clipped = np.clip(combined, 0, 1)
                # Scale the values to the range [0, 255] for PIL
                scaled_values = (combined_clipped * 255).astype(np.uint8)
                # Convert the numpy array to a PIL Image
                img = Image.fromarray(scaled_values, mode='L')
                # Define the filename based on your variables
                filename_inv = f"{label}_{base_face}_{seed:01d}_{trial_adj:05d}_p_inv.png"
                # Save the image as a PNG file
                img.save(os.path.join(stimulus_path, filename_inv))
    
    if save_rdata:
        timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M")
        filename_rdata = f"{label}_seed{seed}_time_{timestamp}"

        base_faces_unfolded = {f"base_faces_{key}": value for key, value in base_faces.items()}
        stimuli_params_unfolded = {f"stimuli_params_{key}": value for key, value in stimuli_params.items()}

        np.savez(os.path.join(stimulus_path, filename_rdata), 
                 patches=p['patches'],
                 patchIdx=p['patchIdx'],
                 noise_type=p['noise_type'],
                 version=p['generator_version'],
                 **base_faces_unfolded,
                 **stimuli_params_unfolded)

    if return_as_dataframe:
        return stimuli