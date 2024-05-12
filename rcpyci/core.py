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

#import jax.numpy as jnp
import numpy as np
from im_ops import combine, get_image_size, save_image
#from jax import jit
from pipelines import (
    ci_postprocessing_pipeline,
    compute_zmap_ttest_pipeline,
    default_ci_postprocessing_pipeline_kwargs,
    default_compute_zmap_ttest_pipeline_kwargs,
)
from tqdm import tqdm

#TODO fix stimuli and params to stimuli be indices and params the params (fixed already?)
#TODO clean up interfaces (filepath not needed)
#TODO add logging

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
        stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed=seed)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)
    if stimuli_order is None:
        stimuli_order = np.arange(0,responses.shape[0]).astype(int)

    # reorder stimuli params based on the selection order
    stimuli_params = stimuli_params[stimuli_order]
    if anti_ci:
        stimuli_params = -stimuli_params
    
    ci = __generate_ci_noise(stimuli_params, responses, patches, patch_idx)
    # combine the base face with the aggregated ci noise image and apply post-processing
    combined = ci_postproc_pipe(base_image, ci, stimuli_params, responses, patches, patch_idx, **ci_postproc_kwargs)
    return ci, combined

def compute_ci_and_zmap(base_image: np.ndarray,
                        responses: np.ndarray,
                        stimuli_order: np.ndarray = None,
                        stimuli_params: np.ndarray = None,
                        ci_postproc_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
                        ci_postproc_kwargs: dict = default_ci_postprocessing_pipeline_kwargs,
                        zmap_pipe: Callable[[Any], Any] = compute_zmap_ttest_pipeline,
                        zmap_kwargs: dict = default_compute_zmap_ttest_pipeline_kwargs,
                        anti_ci: bool = False,
                        n_trials: int = 770,
                        n_scales: int = 5,
                        sigma: int = 5,
                        noise_type: str = 'sinusoid',
                        save_ci: bool = True,
                        save_zmap: bool = True,
                        path: str = './zmaps',
                        label: str = 'experiment',
                        seed: int = 1):
    img_size = get_image_size(base_image)
    # Load parameter file (created when generating stimuli)
    stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed=seed)
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, n_scales=n_scales, noise_type=noise_type, sigma=sigma)

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
        zmap = zmap_pipe(base_image, ci, stimuli_params, responses, patches, patch_idx, **zmap_kwargs)
        np.save("/home/hval/rcpyci/zmap/zmap.npy", zmap)
    #del stimuli_params, responses, patches, patch_idx

    return ci, zmap

# average out individual responses to create an aggregate ci
def __generate_ci_noise(stimuli, responses, patches, patch_idx):
    weighted = stimuli * responses
    if weighted.ndim == 1:
        params = weighted
    else:
        params = weighted.mean(axis=0)
    return __generate_individual_noise_stimulus(params, patches, patch_idx)

#@jit
def __generate_individual_noise_stimulus(params, patches, patch_idx):
    # we need to convert to int and subtract 1 to make it 0-indexed
    patch_indices = patch_idx.astype(int)
    pd = patches.shape
    patch_params = np.array(params[patch_indices]).reshape(pd)
    reshaped_matrix = (patches * patch_params).reshape((pd[0]*pd[1], pd[2]))
    noise = np.mean(reshaped_matrix, axis=1).reshape(pd[0:2])
    return noise

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

#@partial(jit, static_argnames=['img_size'])
def __generate_scales(img_size:int = 512):
    x, y = np.meshgrid(np.arange(start=1, stop=img_size+1, step=1), np.arange(start=1, stop=img_size+1, step=1))
    patch_size_int = np.round(x / y).astype(int)
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
# creating the 2IFC. It is important to set the seed for both random and numpy explicitly here
# to ensure reproducibility between experiments. This would generate the same
# parameters for generating stimuli and recreate those for creating the ci
def __generate_stimuli_params(n_trials: int, n_scales: int, seed: int = 1):
    np.random.seed(seed)
    random.seed(seed)
    nparams = sum(6 * 2 * np.power(2, np.arange(n_scales))**2)
    stimuli_params = np.random.uniform(-1, 1, size=(n_trials, nparams))
    return stimuli_params

def __generate_all_noise_stimuli(n_trials, n_scales, img_size, noise_type, sigma, seed):
    stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed)
    stimuli = np.zeros((n_trials, img_size, img_size))
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)
    for trial in tqdm(range(n_trials)):
        params = stimuli_params[trial]
        stimuli[trial,:,:] = __generate_individual_noise_stimulus(params, patches, patch_idx)
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
    os.makedirs(save_path, exist_ok=True)
    img_size = get_image_size(base_face)

    stimuli = __generate_all_noise_stimuli(n_trials, n_scales, img_size, noise_type, sigma, seed)
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
