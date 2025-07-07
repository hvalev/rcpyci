"""
File Name: interface.py
Description: TBD
"""

import logging
import os
from datetime import datetime
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .core import generate_stimuli_2ifc, process_pipelines
from .im_ops import read_image, save_image
from .pipelines import full_pipeline
from .utils import extract_sorted_responses_condition, extract_sorted_responses_participant, pop_consumed_variables

logging.basicConfig(level=logging.INFO)

def process_condition(condition,
                      data: pd.DataFrame,
                      base_image: np.ndarray,
                      pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                      stimuli_params: np.ndarray = None,
                      patches: np.ndarray = None,
                      patch_idx: np.ndarray = None,
                      n_trials: int = 770,
                      n_scales: int = 5,
                      gabor_sigma: int = 5,
                      noise_type: str = 'sinusoid',
                      seed: int = 1,
                      experiment_path: str = './experiment',
                      label:str='rcpyci',
                      return_results: bool = True):
    """
    This function processes a single condition from the given data. 
    It calculates the average response for each stimulus_id across 
    participants within the specified condition. It then uses these 
    responses to generate visual stimuli and apply processing pipelines.

    Args:

        condition (int): The ID of the condition to process.
        data (pd.DataFrame): The dataset containing participant responses.
        base_image (np.ndarray): The base image used for generating visual stimuli.
        pipelines (List[Tuple[Callable, dict]]): A list of processing pipelines to apply.
        stimuli_params (np.ndarray): Optional parameters for generating visual stimuli.
        patches (np.ndarray): Optional patch data.
        patch_idx (np.ndarray): Optional index values for the patches.
        n_trials (int): The number of trials to simulate.
        n_scales (int): The number of scales to use in processing pipelines.
        gabor_sigma (int): The standard deviation of Gabor filters used in processing pipelines.
        noise_type (str): The type of noise to add to the processed stimuli.
        seed (int): A random seed for reproducing results.
        experiment_path (str): The path where the experiment data will be stored.
        label (str): A label identifier for the condition.

    Returns:
        results: A dictionary containing the processed visual stimuli and their corresponding responses.
    """
    kwargs = pop_consumed_variables(['condition', 'data', 'experiment_path', 'label'], locals())

    filtered_data, sorted_responses = extract_sorted_responses_condition(data, condition=condition, n_trials=n_trials)
    kwargs['responses'] = sorted_responses
    
    logging.info(f"Processing condition {condition}")
    pipeline_id = f"{label}_{condition}"
    filtered_data.to_csv(f"{os.path.join(experiment_path,pipeline_id)}_data.csv")
    result = process_pipelines(base_image = base_image,
                        responses = sorted_responses,
                        pipelines = pipelines,
                        pipeline_id = pipeline_id,
                        experiment_path = experiment_path,
                        stimuli_params = stimuli_params,
                        patches = patches,
                        patch_idx = patch_idx,
                        n_trials = n_trials,
                        n_scales = n_scales,
                        gabor_sigma = gabor_sigma,
                        noise_type = noise_type,
                        seed = seed)
    if return_results:
        return result

def process_conditions(conditions, 
                       data: pd.DataFrame,
                       base_image: np.ndarray,
                       pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                       stimuli_params: np.ndarray = None,
                       patches: np.ndarray = None,
                       patch_idx: np.ndarray = None,
                       n_trials: int = 770,
                       n_scales: int = 5,
                       gabor_sigma: int = 25,
                       noise_type: str = 'sinusoid',
                       seed: int = 1,
                       experiment_path:str='./experiment',
                       label:str='rcpyci',
                       n_jobs=10,
                       return_results: bool = False):
    """
    Process conditions for a series of experiments.

    This function processes each condition in the given `conditions` DataFrame,
    applying various pipelines and transformations to generate ci and zmaps.
    The results are returned as a list of processed condition objects.

    Parameters:
        conditions (pd.DataFrame): A DataFrame containing the experiment
            conditions, with columns for the condition names.
        data (pd.DataFrame): Additional data required for processing.
        base_image (np.ndarray): The base image used for processing.
        pipelines (List[Tuple[Callable, dict]]): A list of tuples containing
            callable functions and their corresponding parameters. Defaults to
            `full_pipeline`.
        stimuli_params (np.ndarray): Optional array of stimulus parameters.
        patches (np.ndarray): Optional array of patch indices.
        patch_idx (np.ndarray): Optional array of patch indices.
        n_trials (int): The number of trials per condition (default: 770).
        n_scales (int): The number of scales to use in processing (default: 5).
        gabor_sigma (int): The Gabor filter sigma value (default: 25).
        noise_type (str): The type of noise to add to the images (default:
            'sinusoid').
        seed (int): The random seed for generating noise and other
            randomness (default: 1).
        experiment_path (str): The path to the experiment directory
            (default: './experiment').
        label (str): A label for the experiment (default: 'rcpyci').
        n_jobs (int): The number of parallel jobs to use when processing
            conditions. Defaults to 10.

    Returns:
        list: A list of processed condition objects, each representing a
            set of ci and zmaps generated from the input data.

    Notes:
        Be mindful that with higher parallel jobs, the progress bar may become
        less accurate.
    """
    kwargs = pop_consumed_variables(['conditions', 'n_jobs'], locals())
    
    logging.info("Started calculating ci and zmaps for conditions. "
                    "Be mindful that with higher parallel jobs the progress bar becomes more inaccurate. "
                    "This may take a while... ")
    Parallel(n_jobs=n_jobs)(delayed(process_condition)(
        condition=condition,
        **kwargs
    ) for condition in tqdm(conditions))

    logging.info("Finished processing ci and zmaps for conditions.")

def process_participant(participant,
                        data: pd.DataFrame,
                        base_image: np.ndarray,
                        pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                        stimuli_params: np.ndarray = None,
                        patches: np.ndarray = None,
                        patch_idx: np.ndarray = None,
                        n_trials: int = 770,
                        n_scales: int = 5,
                        gabor_sigma: int = 25,
                        noise_type: str = 'sinusoid',
                        seed: int = 1,
                        experiment_path:str='./experiment',
                        label:str='rcpyci',
                        return_results: bool = True):
    """ 
    Process participant data and apply visual processing pipelines.

    This function takes in a participant ID, along with various data frames, 
    arrays, and other inputs related to the experiment. It processes the 
    participant's responses, applies a series of visual processing pipelines, 
    and returns the results.

    Parameters:
        participant (int): The ID of the participant being processed.
        data (pd.DataFrame): A DataFrame containing the participant's responses.
        base_image (np.ndarray): A 2D array representing the base image used in the experiment.
        pipelines (List[Tuple[Callable, dict]]): A list of tuples, where each tuple 
                    contains a callable function and its arguments. These functions represent 
                    the visual processing pipelines to be applied.
        stimuli_params (np.ndarray): An optional array containing parameters related to the 
                    stimuli presented during the experiment.
        patches (np.ndarray): An optional array containing patch-level data from the participant's responses.
        patch_idx (np.ndarray): An optional array indexing the patches in the patches array.
        n_trials (int, default=770): The number of trials in the experiment.
        n_scales (int, default=5): The number of scales used for constructing the noise patches.
        gabor_sigma (int, default=25): The standard deviation of the Gabor filter.
        noise_type (str, default='sinusoid'): The type of noise to be added to the images.
        seed (int, default=1): A random seed for reproducibility.
        return_results: set to true as if this method is called directly, it is with the intent to 
            return results. Otherwise it's for precomputing and caching
    
    Returns:
        results: A dictionary containing the processed participant data and pipeline results.
    """
    kwargs = pop_consumed_variables(['participant', 'data', 'experiment_path', 'label'], locals())
    
    filtered_data, sorted_responses = extract_sorted_responses_participant(data, participant=participant, n_trials=n_trials)

    kwargs['responses'] = sorted_responses

    logging.info(f"Processing participant {participant}")
    pipeline_id = f"{label}_{participant}"
    
    # Dump participant data for easier lookup later
    filtered_data.to_csv(f"{os.path.join(experiment_path,pipeline_id)}_data.csv")

    results = process_pipelines(base_image = base_image,
                        responses = sorted_responses,
                        pipelines = pipelines,
                        pipeline_id = pipeline_id,
                        experiment_path = experiment_path,
                        stimuli_params = stimuli_params,
                        patches = patches,
                        patch_idx = patch_idx,
                        n_trials = n_trials,
                        n_scales = n_scales,
                        gabor_sigma = gabor_sigma,
                        noise_type = noise_type,
                        seed = seed)
    if return_results:
        return results

def process_participants(participants: list,
                         data: pd.DataFrame,
                         base_image: np.ndarray,
                         pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                         stimuli_params: np.ndarray = None,
                         patches: np.ndarray = None,
                         patch_idx: np.ndarray = None,
                         n_trials: int = 770,
                         n_scales: int = 5,
                         gabor_sigma: int = 25,
                         noise_type: str = 'sinusoid',
                         seed: int = 1,
                         experiment_path:str='./experiment',
                         label: str='rcpyci',
                         n_jobs=10,
                         return_results: bool = False):
    """
    This function processes individual participants, calculating CI and zmaps for each.

    Parameters:
        participants (list): A list of participant IDs.
        data (pd.DataFrame): The raw response data.
        base_image (np.ndarray): The base image used for visualization.
        stimuli_params (np.ndarray, optional): Additional parameters related to the stimuli. Defaults to None.
        n_trials (int, optional): The number of trials used in the analysis. Defaults to 770.
        n_scales (int, optional): The number of scales used in the analysis. Defaults to 5.
        sigma (int, optional): The standard deviation used in the noise generation. Defaults to 5.
        noise_type (str, optional): The type of noise to generate. Defaults to 'sinusoid'.
        seed (int, optional): The random seed for generating noise. Defaults to 1.
        experiment_path (str, optional): The path where the results will be saved. Defaults to './experiment'.
        label (str, optional): A unique label for the experiment. Defaults to 'rcpyci'.
        n_jobs (int): The number of jobs to be used. Defaults to 10.

    Returns:
        result (list): A list of participant IDs with corresponding CI and zmap data.

    """
    kwargs = pop_consumed_variables(['participants', 'n_jobs'], locals())

    logging.info("Started calculating ci and zmaps for individual participants.  " +
                  "Be mindful that with higher parallel jobs the progress bar becomes more inaccurate.  " +
                    "This may take a while... ")
    Parallel(n_jobs=n_jobs)(delayed(process_participant)(participant=participant, **kwargs) for participant in tqdm(participants))
    logging.info("Finished processing individual ci and zmaps.")

def analyze_data(data: pd.DataFrame,
                 base_face_path: str,
                 pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                 stimuli_params: np.ndarray = None,
                 patches: np.ndarray = None,
                 patch_idx: np.ndarray = None,
                 n_trials: int = 770,
                 n_scales: int = 5,
                 gabor_sigma: int = 25,
                 noise_type: str = 'sinusoid',
                 seed: int = 1,
                 experiment_path:str='./experiment',
                 label:str='rcpyci',
                 n_jobs=10,
                 return_results: bool = False):
    """ 
    Analyzes data and generates stimuli parameters and patches for all participants and conditions.

    This function takes in a Pandas DataFrame containing participant data, as well as various 
    input parameters related to image processing and experimental design. It uses these inputs 
    to generate pre-computed values (stimuli parameters and patches) that can be used by subsequent 
    processing pipelines.
    The function first loads the base face image and calculates its size. It then generates stimuli 
    parameters for all participants and conditions using the provided number of trials, scales, 
    and seed value. Similarly, it generates noise patterns for all participants and conditions based 
    on the input image size, number of scales, noise type, and Gabor sigma.
    The function then creates a directory for storing experimental data and passes along input variables 
    (apart from those that have already been consumed) to propagate pre-computed values to subsequent 
    processing pipelines. It also extracts unique participant IDs and condition labels from the input 
    DataFrame.
    Finally, the function calls two separate processing functions - one for participants and one for 
    conditions - passing in the pre-computed values and other relevant inputs. The results of these 
    processes are returned as a tuple containing the participant-level results and condition-level results.

    Parameters:
        data: A Pandas DataFrame containing participant data.
        base_face_path: Path to the base face image file.
        pipelines: Optional list of processing pipelines (default: full_pipeline).
        stimuli_params, patches, patch_idx: Pre-computed values for stimuli parameters and noise patterns (default: None).
        n_trials, n_scales, gabor_sigma: Input parameters for generating stimuli parameters and noise patterns.
        noise_type: Type of noise to generate (default: 'sinusoid').
        seed: Random seed value for generating pre-computed values (default: 1).
        experiment_path: Directory path for storing experimental data (default: './experiment').
        label: Label for the experiment (default: 'rcpyci').
        n_jobs: Number of jobs to run in parallel (default: 10).
    Returns:
        A tuple containing participant-level results and condition-level results. 
    """
    from .core import __generate_noise_pattern, __generate_stimuli_params
    from .im_ops import get_image_size
    
    base_image = read_image(os.path.join(os.getcwd(), base_face_path), grayscale=True)
    
    # calculate stimuli params and patches for all participants and conditions
    # and simply pass them along to propagate the pre-computed values here
    # so that they are not re-computed in core.process_pipelines
    img_size = get_image_size(base_image)
    if stimuli_params is None:
        stimuli_params = __generate_stimuli_params(n_trials, n_scales, seed=seed)
    if patches is None or patch_idx is None:
        patches, patch_idx = __generate_noise_pattern(img_size=img_size, n_scales=n_scales, noise_type=noise_type, gabor_sigma=gabor_sigma)
    
    os.makedirs(experiment_path, exist_ok=True)

    kwargs = pop_consumed_variables(['base_face_path', 'img_size', '__generate_stimuli_params', '__generate_noise_pattern', 'get_image_size'], locals())
    
    participants = list(data['participant_id'].unique())
    kwargs['participants'] = participants
    process_participants(**kwargs)
    kwargs.pop('participants', None)

    conditions = list(data['condition'].unique())
    kwargs['conditions'] = conditions
    process_conditions(**kwargs)


def setup_experiment(base_face_path: str,
                     n_trials: int = 770,
                     n_scales: int = 5,
                     gabor_sigma: int = 25,
                     noise_type: str = 'sinusoid',
                     experiment_path: str = './experiment',
                     label: str = 'rcpyci',
                     seed: int = 1):
    """
    Setup and prepare the experiment for a given base face, number of trials,
    scales, sigma, noise type, and path.

    This function generates the required stimulus material using the provided
    parameters. It also creates folders to save the generated stimuli and data
    files. The timestamp is used to create unique filenames for each trial.
    Finally, it saves the base face image and the experiment configuration as a
    numpy array to disk.

    Parameters:
        base_face_path (str): Path to the base face image file.
        n_trials (int): Number of trials in the experiment. Default is 770.
        n_scales (int): Number of scales used for stimulus generation.
            Default is 5.
        gabor_sigma (int): Sigma value used for Gaussian noise generation. 
            It is inly used when noise_type is 'gabor'. Default is 5. 
        noise_type (str): Type of noise to generate. Default is 'sinusoid'.
            Another option is 'gabor'.
        experiment_path (str): Path where the experiment data will be saved.
            Default is './experiment'.
        label (str): Label for the experiment, used in file names and data
            saving. Default is 'rcpyci'.
        seed (int): Seed value for random number generation. Default is 1.

    Returns:
        None
    """
    
    base_image = read_image(os.path.join(os.getcwd(), base_face_path), grayscale=True)
    assert base_image.shape[0] == base_image.shape[1]

    logging.info("Generating stimulus material")
    stimuli_ori, stimuli_inv = generate_stimuli_2ifc(base_face=base_image,
                                                     n_trials=n_trials,
                                                     n_scales=n_scales,
                                                     gabor_sigma=gabor_sigma,
                                                     noise_type=noise_type,
                                                     seed=seed)
    
    logging.info("Creating folders and saving data to disk")
    os.makedirs(os.path.join(experiment_path), exist_ok=True)
    timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M")
    for trial, (stimulus, stimulus_inverted) in tqdm(enumerate(zip(stimuli_ori, stimuli_inv, strict=False)), desc="Processing", total=len(stimuli_ori)):
        filename_ori = f"stimulus_{label}_seed_{seed}_trial_{trial:0{len(str(n_trials))}d}_ori.png"
        save_image(stimulus, os.path.join(experiment_path,"stimuli",filename_ori))
        filename_inv = f"stimulus_{label}_seed_{seed}_trial_{trial:0{len(str(n_trials))}d}_inv.png"
        save_image(stimulus_inverted, os.path.join(experiment_path,"stimuli",filename_inv))
    
    logging.info("Creating folders and saving data to disk")
    data_path = f"data_{label}_seed_{seed}_time_{timestamp}"
    np.savez(os.path.join(experiment_path, data_path), 
             base_face_path=base_face_path,
             n_trials=n_trials,
             n_scales=n_scales,
             gabor_sigma=gabor_sigma,
             noise_type=noise_type,
             experiment_path=experiment_path,
             label=label,
             seed=seed)
    logging.info("Done!")

