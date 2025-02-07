import os
import random
from datetime import datetime

import numpy as np
from generate_noise import generate_noise_pattern
from generateNoiseImage import generate_noise_image
from PIL import Image
from tqdm import tqdm


def read_image(filename, grayscale=False):
    img = Image.open(filename)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    return np.asarray(img)

def generate_stimuli_2IFC(base_face_files, overwrite_params=False, params_values=None, n_trials=770, img_size=512, stimulus_path='./stimuli', label='rcic', use_same_parameters=True, seed=1, maximize_baseimage_contrast=True, noise_type='sinusoid', nscales=5, sigma=25, ncores=None, return_as_dataframe=False, save_as_png=True, save_rdata=True):
    
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
        # Example usage:
        #filename = 'example.jpg'
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
    
    # Generate noise pattern
    p = generate_noise_pattern(img_size=img_size, noise_type=noise_type, nscales=nscales, sigma=sigma)

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
    
    # Save all to image file
    # generator_version = '0.4.0'
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

        # params = npzfile['s'] if 's' in npzfile else npzfile['p']['patches']
        # base_faces = npzfile['base_faces']
        # stimuli_params = npzfile['stimuli_params']
        #np.savez(os.path.join(stimulus_path, filename_rdata), **locals(), allow_pickle=True)
    
    if return_as_dataframe:
        return stimuli