"""
File Name: core.py
Description: TBD
"""
import math
import os
import random
from datetime import datetime
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from im_ops import combine, get_image_size, read_image, save_image
from jax import jit
from PIL import Image
from pipelines import ci_postprocessing_pipeline, default_ci_postprocessing_pipeline_kwargs
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_1samp
from tqdm import tqdm

#TODO better function names
#TODO fix stimuli and params to stimuli be indices and params the params (fixed already?)
#TODO clean up interfaces (filepath not needed)
#TODO add logging
#TODO make sure there is a global way to set seed
    
def compute_ci(base_image: np.ndarray,
               responses: np.ndarray,
               stimuli_order: np.ndarray = None,
               stimuli_params: np.ndarray = None,
               patches: np.ndarray = None,
               patch_idx: np.ndarray = None,
               ci_postproc_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
               ci_postproc_kwargs: dict = default_ci_postprocessing_pipeline_kwargs,
               anti_ci: bool = False,
               n_trials: int = 770,
               n_scales: int = 5,
               sigma: int = 5,
               noise_type: str = 'sinusoid',
               seed=1):
    img_size = get_image_size(base_image)
    
    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials, n_scales)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)
    if stimuli_order is None:
        stimuli_order = np.arange(0,responses.shape[0]).astype(int)

    # reorder stimuli params based on the selection order
    stimuli_params = stimuli_params[stimuli_order]
    if anti_ci:
        stimuli_params = -stimuli_params
    
    ci = generate_ci_noise(stimuli_params, responses, patches, patch_idx)
    # combine the base face with the aggregated ci noise image and apply post-processing
    combined = ci_postproc_pipe(base_image, ci, **ci_postproc_kwargs)
    return ci, combined

def compute_ci_and_zmap(base_image: np.ndarray,
                        responses: np.ndarray,
                        stimuli_order: np.ndarray = None,
                        stimuli_params: np.ndarray = None,
                        ci_postproc_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
                        ci_postproc_kwargs: dict = default_ci_postprocessing_pipeline_kwargs,
                        zmap_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
                        zmap_kwargs: dict = default_ci_postprocessing_pipeline_kwargs,
                        anti_ci: bool = False,
                        n_trials: int = 770,
                        n_scales: int = 5,
                        sigma:int = 5,
                        noise_type='sinusoid',
                        save_ci=True,
                        save_zmap=True,
                        zmap_method='t.test',
                        threshold=3,
                        zmaptargetpath='./zmaps',
                        label='experiment',
                        seed=1):
    img_size = get_image_size(base_image)
    # Load parameter file (created when generating stimuli)
    stimuli_params = __generate_stimuli_params(n_trials, n_scales)
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)

    ci, combined = compute_ci(base_image = base_image,
                    responses = responses,
                    stimuli_order = stimuli_order,
                    stimuli_params = stimuli_params,
                    patches = patches,
                    patch_idx= patch_idx,
                    ci_postproc_pipe = ci_postproc_pipe,
                    ci_postproc_kwargs = ci_postproc_kwargs,
                    anti_ci = anti_ci,
                    n_trials= n_trials,
                    n_scales= n_scales,
                    sigma = sigma,
                    noise_type = noise_type,
                    seed = seed)
    

    filename = ''
    if anti_ci:
        filename += 'antici_' + label + ".png"
        stimuli_params = -stimuli_params
    else:
        filename += 'ci_' + label + ".png"

    if save_ci:
        save_image(image=combined, path='/home/hval/rcpyci/cis/'+filename)

    zmap = None
    if save_zmap:
        if zmap_method == 'quick':
            zmap = process_quick_zmap(ci, sigma, threshold)
        elif zmap_method == 't.test':
            zmap = process_ttest_zmap(stimuli_params, responses, patches, patch_idx, img_size, ci)
            #TODO remove this and pull the save in the aggregated function
            np.save("/home/hval/rcpyci/zmap/zmap.npy", zmap)
        else:
            raise ValueError(f"Invalid zmap method: {zmap_method}")
    return ci, zmap

def process_quick_zmap(ci, sigma, threshold):
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    zmap = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return zmap

def process_ttest_zmap(params, responses, patches, patch_idx, img_size, ci):
    weighted_parameters = params * responses
    n_observations = len(responses)
    noise_images = np.zeros((img_size, img_size, n_observations))
    for obs in range(n_observations):
        noise_images[:, :, obs] = __generate_noise_image(weighted_parameters[obs], patches, patch_idx)
    t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    zmap = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
    return zmap

# average out individual responses to create an aggregate ci
def generate_ci_noise(stimuli, responses, patches, patch_idx):
    weighted = stimuli * responses
    if weighted.ndim == 1:
        params = weighted
    else:
        params = weighted.mean(axis=0)
    return __generate_noise_image(params, patches, patch_idx)

#TODO This is jittable and equivalent with jnp equivalent
#@jit
def __generate_noise_image(params, patches, patch_idx):
    # we need to convert to int and subtract 1 to make it 0-indexed
    patch_indices = patch_idx.astype(int)
    pd = patches.shape
    patch_params = np.array(params[patch_indices]).reshape(pd)
    reshaped_matrix = (patches * patch_params).reshape((pd[0]*pd[1], pd[2]))
    noise = np.mean(reshaped_matrix, axis=1).reshape(pd[0:2])
    return noise

#TODO better name
def __generate_coordinate_meshgrid_for_patch(cycles, patch_size):
    x = np.linspace(0, cycles, patch_size)
    y = np.linspace(0, cycles, patch_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def __generate_sinusoid(patch_size: int, cycles: float, angle: float, phase: float, contrast: float):
    X, Y = __generate_coordinate_meshgrid_for_patch(cycles, patch_size)
    angle = math.radians(angle)
    sinepatch = X * math.cos(angle) + Y * math.sin(angle)
    sinusoid = (sinepatch * 2 * math.pi) + phase
    sinusoid = contrast * np.sin(sinusoid)
    return sinusoid

def __generate_gabor(patch_size, cycles, angle, phase, sigma, contrast):
    sinusoid = __generate_sinusoid(patch_size, cycles, angle, phase, contrast)
    x0 = np.linspace(-0.5, 0.5, patch_size)
    X, Y = np.meshgrid(x0, x0)
    gauss_mask = np.exp(-((X ** 2 + Y ** 2) / (2 * (sigma / patch_size) ** 2)))
    gabor = gauss_mask * sinusoid
    return gabor

@partial(jit, static_argnames=['img_size'])
def __generate_scales(img_size:int = 512):
    x, y = jnp.meshgrid(jnp.arange(start=1, stop=img_size+1, step=1), jnp.arange(start=1, stop=img_size+1, step=1))
    patch_size_int = jnp.round(x / y).astype(int)
    return patch_size_int

def __generate_noise_pattern(img_size=512, n_scales=5, noise_type='sinusoid', sigma=25):
    orientations = np.array([0, 30, 60, 90, 120, 150])
    phases = np.array([0, np.pi/2])
    scales = 2 ** np.arange(n_scales)
    assert scales.dtype == np.int64

    patch_sizes = __generate_scales(img_size=img_size)
    nr_patches = len(scales) * len(orientations) * len(phases)

    patches = np.zeros((img_size, img_size, nr_patches))
    patch_idx = np.zeros((img_size, img_size, nr_patches))

    co = 0
    idx = 0

    for scale in scales:
        # iterate over each scale (i.e. 512, 256, 128, 64, 32)
        patch = patch_sizes[scale - 1, img_size - 1]
        for orientation in orientations:
            for phase in phases:
                if noise_type == 'gabor':
                    p = __generate_gabor(patch, 1.5, orientation, phase, sigma, 1)
                else:
                    p = __generate_sinusoid(patch, 2, orientation, phase, 1)
                # Repeat to fill scale
                patches[:, :, co - 1] = np.tile(p, (scale, scale))
                for col in range(1, scale + 1):
                    for row in range(1, scale + 1):
                        patch_idx[patch * (row - 1) : patch * row, patch * (col - 1):patch * col, co - 1] = idx
                        # Update contrast counter
                        idx += 1
                co += 1
    return patches, patch_idx

# this function creates a uniform distribution centered at 0
# the purpose is to generate the stimulus parameters used for 
# creating the 2IFC. The most important thing for reproducibility
# is to set the seed for random and numpy so that the results are 
# reproducible
def __generate_stimuli_params(n_trials: int, n_scales: int, seed: int = 1):
    nparams = sum(6 * 2 * np.power(2, np.arange(n_scales))**2)
    stimuli_params = np.random.uniform(-1, 1, size=(n_trials, nparams))
    return stimuli_params

def __generate_stimuli_noise(n_trials, n_scales, img_size, noise_type, sigma):
    stimuli_params = __generate_stimuli_params(n_trials, n_scales)
    stimuli = np.zeros((n_trials, img_size, img_size))
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)
    for trial in tqdm(range(n_trials)):
        params = stimuli_params[trial]
        noise_pattern = __generate_noise_image(params, patches, patch_idx)
        stimuli[trial,:,:] = noise_pattern
    return stimuli

def __generate_stimulus_image(stimulus, base_face):
    stimulus = (stimulus + 0.3) / 0.6
    return combine(stimulus, base_face)

def generate_stimuli_2IFC(base_face: np.ndarray,
                          n_trials:int = 770,
                          n_scales:int=5,
                          sigma:int=5,
                          noise_type:str='sinusoid',
                          save_path:str='./stimuli',
                          label='rcpyci',
                          seed=1):
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(save_path, exist_ok=True)
    img_size = get_image_size(base_face)

    stimuli = __generate_stimuli_noise(n_trials, n_scales, img_size, noise_type, sigma)
    assert n_trials == stimuli.shape[0]

    for trial in tqdm(range(0,n_trials)):
        stimulus = __generate_stimulus_image(stimuli[trial, :, :], base_face)
        filename_ori = f"stimulus_{label}_seed_{seed}_trial_{trial:0{len(str(n_trials))}d}_ori.png"
        save_image(stimulus, os.path.join(save_path,filename_ori))
        stimulus_inverted = __generate_stimulus_image(-stimuli[trial, :, :], base_face)
        filename_inv = f"stimulus_{label}_seed_{seed}_trial_{trial:0{len(str(n_trials))}d}_inv.png"
        save_image(stimulus_inverted, os.path.join(save_path,filename_inv))

    timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M")
    data = f"data_{label}_seed_{seed}_time_{timestamp}"
    np.savez(os.path.join(save_path, data), 
             base_face=base_face,
             n_trials=n_trials,
             n_scales=n_scales,
             sigma=sigma,
             noise_type=noise_type,
             save_path=save_path,
             label=label,
             seed=seed)
    return stimuli

# Usage
from pipelines import default_stimuli_generation_kwargs

base_image = read_image(os.getcwd()+"/rcpyci/"+"base_face.jpg", grayscale=True)
result_python_unconv = generate_stimuli_2IFC(base_face=base_image,
                                             **default_stimuli_generation_kwargs)

print(result_python_unconv.shape)
n_trials = default_stimuli_generation_kwargs['n_trials']
stimuli = np.arange(n_trials)
responses = np.ones(shape=(n_trials,1))
# Define your arguments
pipeline_kwargs = {
    'scaling': 'independent',
    'scaling_constant': 0.1,
    'mask': None
}

ci, zmap = compute_ci_and_zmap(base_image=base_image,
                            stimuli_order=stimuli,
                            responses=responses,
                            **default_stimuli_generation_kwargs,
                            ci_postproc_pipe=ci_postprocessing_pipeline,
                            ci_postproc_kwargs=pipeline_kwargs,
                            anti_ci=False,
                            save_ci=True,
                            save_zmap=True,
                            zmap_method='t.test',
                            threshold=1)



def compare_images(img1_path, img2_path):
    # Open images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Check if images have the same size
    if img1.size != img2.size:
        print("Images have different dimensions.")
        return False

    # Convert images to grayscale if needed
    img1 = img1.convert("L")
    img2 = img2.convert("L")

    # Compare pixel values
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            if img1.getpixel((x, y)) != img2.getpixel((x, y)):
                print("Images are not equivalent.")
                return False

    # If no differing pixels are found, images are equivalent
    print("Images are equivalent.")
    return True

def compare_images_in_folders(folder1, folder2):
    # List PNG files in both folders
    png_files1 = [file for file in os.listdir(folder1) if file.endswith('.png')]
    png_files2 = [file for file in os.listdir(folder2) if file.endswith('.png')]

    # Iterate over PNG files in folder1
    for file1 in png_files1:
        # Check if there is an equivalent file in folder2
        file2 = file1
        if file2 in png_files2:
            img1_path = os.path.join(folder1, file1)
            img2_path = os.path.join(folder2, file2)
            print(f"Comparing {file1} and {file2}:")
            if not compare_images(img1_path, img2_path):
                return False
        else:
            print(f"No equivalent file found for {file1} in folder2.")

# Paths to the folders containing PNG files
folder1 = "stimuli_ref"
folder2 = "stimuli"

# Compare images in folders
compare_images_in_folders(folder1, folder2)

print("COMPARING GENERATED CI")
compare_images('/home/hval/rcpyci/cis/python_ci_compared.png','/home/hval/rcpyci/cis/ci_experiment.png')

test = np.load("/home/hval/rcpyci/zmap/zmap_comp.npy")
if np.array_equal(zmap, test):
    print('ZMAP t.test are equal also')
elif np.allclose(zmap, test):
    a = zmap - test
    print(f'ZMAP t.test are allclose with max {np.max(a)} and min {np.min(a)}')
else:
    raise KeyError