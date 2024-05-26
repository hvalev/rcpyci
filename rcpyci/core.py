"""
File Name: core.py
Description: object to object part of the library, for any interaction and interfacing with files, check interface.py
"""
import math
import random
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .im_ops import combine, get_image_size
from .pipelines import (
    ci_postprocessing_pipeline,
    ci_postprocessing_pipeline_kwargs,
    compute_zmap_ttest_pipeline,
    compute_zmap_ttest_pipeline_kwargs,
)


def compute_ci(base_image: np.ndarray,
               responses: np.ndarray,
               stimuli_params: np.ndarray = None,
               patches: np.ndarray = None,
               patch_idx: np.ndarray = None,
               ci_postproc_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
               ci_postproc_kwargs: dict = ci_postprocessing_pipeline_kwargs,
               anti_ci: bool = False,
               n_trials: int = 770,
               n_scales: int = 5,
               sigma: int = 5,
               noise_type: str = 'sinusoid',
               seed: int = 1):
    """
    Compute the classification image for a given responses to a set of stimuli.

    Args:
        base_image (np.ndarray): The base image used for computation.
        responses (np.ndarray): The response values corresponding to each stimulus.
        stimuli_params (np.ndarray, optional): The parameters of the stimuli. Defaults to None.
        patches (np.ndarray, optional): The noise patterns used for computation. Defaults to None.
        patch_idx (np.ndarray, optional): The indices corresponding to the patches. Defaults to None.
        ci_postproc_pipe (Callable[[Any], Any], optional): The post-processing pipeline for CI computation. Defaults to ci_postprocessing_pipeline.
        ci_postproc_kwargs (dict, optional): The keyword arguments for the post-processing pipeline. Defaults to default_ci_postprocessing_pipeline_kwargs.
        anti_ci (bool, optional): Whether to generate the confidence intervals as the opposite of the true values. Defaults to False.
        n_trials (int, optional): The number of trials used for computation. Defaults to 770.
        n_scales (int, optional): The number of scales used for noise generation. Defaults to 5.
        sigma (int, optional): The standard deviation used for noise generation. Defaults to 5.
        noise_type (str, optional): The type of noise used for generation. Defaults to 'sinusoid'.
        seed (int, optional): The random seed used for generating the noise patterns. Defaults to 1.

    Returns:
        tuple: A tuple containing the classification image and it combined with the base image.
    """
    img_size = get_image_size(base_image)

    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed=seed)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, sigma=sigma)

    if anti_ci:
        stimuli_params = -stimuli_params
    
    # ci = 
    return __generate_ci_noise(stimuli_params, responses, patches, patch_idx)
    # combine the base face with the aggregated ci noise image and apply post-processing
    combined = ci_postproc_pipe(base_image, ci, stimuli_params, responses, patches, patch_idx, **ci_postproc_kwargs)
    return ci, combined

def compute_ci_and_zmap(base_image: np.ndarray,
                        responses: np.ndarray,
                        stimuli_params: np.ndarray = None,
                        ci_postproc_pipe: Callable[[Any], Any] = ci_postprocessing_pipeline,
                        ci_postproc_kwargs: dict = ci_postprocessing_pipeline_kwargs,
                        zmap_pipe: Callable[[Any], Any] = compute_zmap_ttest_pipeline,
                        zmap_kwargs: dict = compute_zmap_ttest_pipeline_kwargs,
                        anti_ci: bool = False,
                        n_trials: int = 770,
                        n_scales: int = 5,
                        sigma: int = 5,
                        noise_type: str = 'sinusoid',
                        seed: int = 1):
    """Compute CI and Z-map: Compute the classification image (CI) and z-map from base image, responses, and optional parameters.

    Parameters:
        base_image (np.ndarray): The base image.
        responses (np.ndarray): The *in-order* responses to stimuli.
        stimuli_params (np.ndarray): Optional parameter for sideloading pre-generated parameter space.
        ci_postproc_pipe (Callable[[Any], Any]): Pipeline for post-processing the CI noise image.
        ci_postproc_kwargs (dict): Keyword arguments for the post-processing pipeline.
        anti_ci (bool): Whether to negate the stimuli parameters.
        n_trials (int): Number of trials in the experiment. Default is 770.
        n_scales (int): Number of scales in the noise pattern. Default is 5.
        sigma (int): Standard deviation of the noise pattern. Default is 5.
        noise_type (str): Type of noise pattern. Default is 'sinusoid'.
        seed (int): Seed for random number generation. Default is 1.

    Returns:
        ci (np.ndarray): The computed CI noise image.
        combined (np.ndarray): The combined base face and aggregated CI noise image, with post-processing applied.
        z_map (np.ndarray): The z-map of the responses to stimuli, if the compute_zmap_ pipeline is provided.

    """
    img_size = get_image_size(base_image)
    # if stimuli_params is passed, no need to recompute it
    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed=seed)
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, n_scales=n_scales, noise_type=noise_type, sigma=sigma)

    ci, combined = compute_ci(base_image = base_image,
                              responses = responses,
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
    
    # Get all local variables, and pipe the ones needed for the postprocessing pipelines, e.g. zmap, infoval, etc
    local_variables = locals()
    piped_local_variables = ['base_image', 'ci', 'stimuli_params', 'responses', 'patches', 'patch_idx', 'anti_ci', 'n_trials', 'n_scales', 'sigma', 'noise_type', 'seed']
    local_variables = {key: local_variables[key] for key in piped_local_variables}

    zmap = None
    if zmap_pipe is not None:
        if anti_ci:        
            local_variables['stimuli_params'] = -stimuli_params
        zmap = zmap_pipe(**local_variables, **zmap_kwargs)
        
    return ci, combined, zmap

def __generate_ci_noise(stimuli, responses, patches, patch_idx):
    """
    Generate CI noise for the given stimuli parameters, responses, patches, and patch indices.

    This function generates weighted CI noise by multiplying the stimuli parameters with the corresponding responses. 
    The resulting weighted CI noise is then processed using the provided post-processing pipeline.

    Parameters:
    - stimuli_params: A 1D NumPy array containing the stimuli parameters.
    - responses: A 2D NumPy array where each row represents a response to the stimuli.
    - patches: A 2D NumPy array representing the noise pattern patches.
    - patch_idx: A 1D NumPy array containing the indices of the noise pattern patches.

    Returns:
    - The generated CI noise as a 2D NumPy array.
    """
    weighted = stimuli * responses
    if weighted.ndim == 1:
        params = weighted
    else:
        params = weighted.mean(axis=0)
    return __generate_individual_noise_stimulus(params, patches, patch_idx)

def __generate_individual_noise_stimulus(params, patches, patch_idx):
    """
    Generate individual noise stimuli for CI computation.

    This function generates individual noise stimuli based on the given parameters.
    It takes in the following inputs:
    - `params`: The weighted stimuli array
    - `patches`: The noise pattern array
    - `patch_idx`: The indices of the patches

    Returns: A 2D array representing the individual noise stimuli
    """
    patch_indices = patch_idx.astype(int)
    pd = patches.shape
    patch_params = params[patch_indices].reshape(pd)
    reshaped_matrix = (patches * patch_params).reshape((pd[0]*pd[1], pd[2]))
    noise = np.mean(reshaped_matrix, axis=1).reshape(pd[:2])
    return noise

def __generate_coordinate_meshgrid_for_patch(cycles, patch_size):
    """
    Generate a coordinate meshgrid for a given patch size.

    Parameters:
    cycles (int): The number of cycles.
    patch_size (int): The size of the patch.

    Returns:
    X (numpy array): The x-coordinates of the meshgrid.
    Y (numpy array): The y-coordinates of the meshgrid.
    """
    x = np.linspace(0, cycles, patch_size)
    y = np.linspace(0, cycles, patch_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def __generate_sinusoid(patch_size: int, cycles: float, angle: float, phase: float, contrast: float):
    """
    Generate a sinusoidal pattern for the noise stimulus.

    Parameters:
    patch_size (int): The size of the patches.
    cycles (float): The number of cycles in the sine wave.
    angle (float): The angle of the sine wave in radians.
    phase (float): The phase shift of the sine wave.
    contrast (float): The contrast level of the sine wave.

    Returns:
    A 2D array representing the sinusoidal pattern.
    """
    X, Y = __generate_coordinate_meshgrid_for_patch(cycles, patch_size)
    angle = math.radians(angle)
    sinepatch = X * math.cos(angle) + Y * math.sin(angle)
    sinusoid = (sinepatch * 2 * math.pi) + phase
    sinusoid = contrast * np.sin(sinusoid)
    return sinusoid

def __generate_gabor(patch_size:int, cycles:float, angle:float, phase:float, gabor_sigma:float, contrast:float):
    """
    This code defines a function that generates Gabor patches with varying parameters.
    The `__generate_gabor` function takes in the patch size, cycles, angle, phase, and contrast as inputs,
    and returns a 2D array representing the Gabor pattern.

    Args:
        patch_size (int): The size of the Gabor patch
        cycles (float): The number of cycles for the sinusoid
        angle (float): The angle of rotation for the sinusoid
        phase (float): The phase shift for the sinusoid
        contrast (float): The amplitude of the sinusoid

    Returns:
        A 2D array representing the Gabor pattern.

    """
    sinusoid = __generate_sinusoid(patch_size, cycles, angle, phase, contrast)
    x0 = np.linspace(-0.5, 0.5, patch_size)
    X, Y = np.meshgrid(x0, x0)
    gauss_mask = np.exp(-((X ** 2 + Y ** 2) / (2 * (gabor_sigma / patch_size) ** 2)))
    return gauss_mask * sinusoid

def __generate_scales(img_size: int = 512):
    """
    Generate scales for an image of size 'img_size'.

    This function is used to generate the scales for a Gabor patch. It returns
    an array where each element corresponds to the scale of a Gabor patch.
    
    Parameters:
    img_size (int): The size of the image.

    Returns:
    An array containing the scales for each Gabor patch.
    """
    x, y = np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1))
    patch_size_int = np.round(x / y).astype(int)
    return patch_size_int

def __generate_noise_pattern(img_size:int=512, n_scales:int=5, noise_type:str='sinusoid', gabor_sigma:float=25):
    """
    Generate a noise pattern for a given image size and number of scales.

    This function generates a noise pattern consisting of Gabor or sinusoid patterns 
    at different orientations, phases, and scales. The resulting noise pattern is 
    used to create stimuli for visual perception experiments.

    Parameters:
    - img_size: The size of the image (default=512).
    - n_scales: The number of scales in the noise pattern (default=5).
    - noise_type: The type of noise pattern to generate ('gabor' or 'sinusoid') 
    (default='sinusoid').
    - sigma: The standard deviation of the Gaussian mask used in the Gabor filter.

    Returns:
    - patches: A 3D array containing the generated noise patterns.
    - patch_idx: A 3D array indexing the different patches in the noise pattern.
    """
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
                    p = __generate_gabor(patch, 1.5, orientation, phase, gabor_sigma, 1)
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

def __generate_stimuli_params(n_trials: int, n_scales: int, seed: int = 1):
    """
    Generate stimulus parameters.

    This function generates a uniform distribution centered at 0, 
    which is used as the stimulus parameters in visual perception experiments. 

    The function takes three parameters: n_trials (the number of trials), 
    n_scales (the number of scales), and seed (the random seed).

    Parameters:
    - n_trials: The number of trials.
    - n_scales: The number of scales.
    - seed: The random seed.

    Returns:
    - stimuli_params: A 2D array containing the generated stimulus parameters.
    """
    # It is important to propagate the seed for both random and numpy explicitly here
    # to ensure reproducibility between experiments. This would generate the exact same
    # parameters used to generate the stimuli and later recreate those for creating the ci
    # without the need to store or load data externally. However, if needed, the parameters
    # can be sideloaded omitting the execution of this function in its caller functions
    np.random.seed(seed)
    random.seed(seed)
    nparams = sum(6 * 2 * np.power(2, np.arange(n_scales))**2)
    stimuli_params = np.random.uniform(-1, 1, size=(n_trials, nparams))
    return stimuli_params

def __generate_all_noise_stimuli(n_trials: int, n_scales: int, img_size: int, noise_type: str, gabor_sigma: float, seed: int) -> np.ndarray:
    """
    Generate all noise stimuli.

    This function generates a set of noise patterns at different orientations, phases, and scales. 
    The resulting noise pattern is used to create stimuli for visual perception experiments.

    Parameters:
        n_trials (int): The number of trials.
        n_scales (int): The number of scales in the noise pattern.
        img_size (int): The size of the image.
        noise_type (str): The type of noise pattern to generate ('gabor' or 'sinusoid').
        gabor_sigma (float): The standard deviation of the Gaussian mask used in the Gabor filter.
        seed (int): The random seed.

    Returns:
        stimuli (np.ndarray): A 3D array containing the generated noise patterns.
    """
    stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed)
    stimuli = np.zeros((n_trials, img_size, img_size))
    patches, patch_idx = __generate_noise_pattern(img_size=img_size, noise_type=noise_type, n_scales=n_scales, gabor_sigma=gabor_sigma)
    for trial in tqdm(range(n_trials), desc="Processing", total=n_trials):
        params = stimuli_params[trial]
        stimuli[trial,:,:] = __generate_individual_noise_stimulus(params, patches, patch_idx)
    return stimuli

def  __generate_stimulus_image(stimulus: np.ndarray, base_face: np.ndarray):
    """
    Generate a stimulus image by combining the noise pattern with a base face.

    This function takes two parameters: 
    - stimulus (np.ndarray): The generated noise pattern.
    - base_face (np.ndarray): The base face to combine with the noise pattern. 

    Returns:
    - The combined stimulus image as an np.ndarray.
    """
    stimulus = (stimulus + 0.3) / 0.6
    return combine(stimulus, base_face)

def generate_stimuli_2IFC(base_face: np.ndarray,
                          n_trials: int = 770,
                          n_scales: int = 5,
                          gabor_sigma: int = 25,
                          noise_type: str = 'sinusoid',
                          seed: int = 1):
    """
    Generate stimuli for a 2IFC (two-interval forced-choice) experiment.

    This function generates two sets of stimuli: one for the "original" interval and another for the "inverted" interval.
    The images in these sets represent pairs of images which need to be presented alongside each other.

    Parameters:
        base_face (np.ndarray): The base face used to generate the stimuli.
        n_trials (int): The number of trials in the experiment. Default is 770.
        n_scales (int): The number of scales in the noise pattern. Default is 5.
        sigma (float): The standard deviation of the Gaussian mask used in the Gabor filter. Default is 5.
        noise_type (str): The type of noise pattern to generate ('gabor' or 'sinusoid'). Default is 'sinusoid'.
        seed (int): The random seed for generating the stimuli.

    Returns:
        A tuple containing two sets of stimulus images: one for the original interval and another for the inverted interval.
    """
    img_size = get_image_size(base_face)

    stimuli = __generate_all_noise_stimuli(n_trials, n_scales, img_size, noise_type, gabor_sigma, seed)
    assert n_trials == stimuli.shape[0]
    stimuli_ori = []
    stimuli_inv = []
    for trial in range(0,n_trials):
        stimulus = __generate_stimulus_image(stimuli[trial, :, :], base_face)
        stimulus_inverted = __generate_stimulus_image(-stimuli[trial, :, :], base_face)
        stimuli_ori.append(stimulus)
        stimuli_inv.append(stimulus_inverted)
    return stimuli_ori, stimuli_inv
