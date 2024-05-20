"""
File Name: interface.py
Description: TBD
"""

import logging
import os
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .core import compute_ci_and_zmap, generate_stimuli_2IFC
from .im_ops import read_image, save_image
from .pipelines import (
    ci_postprocessing_pipeline,
    ci_postprocessing_pipeline_kwargs,
    compute_zmap_ttest_pipeline,
    compute_zmap_ttest_pipeline_kwargs,
)
from .utils import create_test_data, skip_if_exist, verify_data

logging.basicConfig(level=logging.INFO)

@skip_if_exist
def process_condition(condition,
                      data: pd.DataFrame,
                      base_image: np.ndarray,
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
                      seed: int = 1,
                      save_ci: bool = True,
                      save_zmap: bool = True,
                      experiment_path: str = './experiment',
                      label:str='rcpyci'):
    # pass along input variables apart from already consumed ones
    kwargs = {k: v for k, v in locals().items()}
    consumed_variables = ['condition', 'data', 'save_ci', 'save_zmap', 'experiment_path', 'label']
    for key in consumed_variables:
        kwargs.pop(key, None)
    # Sort the dataframe by 'condition', 'participant_id', and 'stimulus_id'
    sorted_data = data.sort_values(by=['condition', 'participant_id', 'stimulus_id'])
    # Calculate the average response for each stimulus_id across participants within each condition
    grouped_data = sorted_data.groupby(['condition', 'stimulus_id'])['responses'].mean().reset_index()
    # extract data for a single condition and sort by stimulus_id
    single_condition_data = grouped_data[grouped_data['condition'] == condition].sort_values(by='stimulus_id')
    sorted_responses = single_condition_data['responses'].to_numpy().reshape(n_trials, 1)
    kwargs['responses'] = sorted_responses
    
    ci, combined, zmap = compute_ci_and_zmap(**kwargs)
    
    ci_filename = f"ci_{label}_{condition}.png"
    if anti_ci:
        ci_filename = f"antici_{label}_{condition}.png"
    
    zmap_filename = f"zmap_{label}_{condition}.png"

    if save_ci:
        save_image(image=ci, path=os.path.join(experiment_path, "ci_raw", ci_filename), save_npy=True)
    if save_ci:
        save_image(image=combined, path=os.path.join(experiment_path, "ci", ci_filename), save_npy=True)
    if zmap_pipe is not None and save_zmap:
        save_image(image=zmap, path=os.path.join(experiment_path, "zmap", zmap_filename), save_npy=True)

    return condition, ci, combined, zmap

def process_conditions(conditions, 
                       data: pd.DataFrame,
                       base_image: np.ndarray,
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
                       seed: int = 1,
                       save_ci: bool = True,
                       save_zmap: bool = True,
                       experiment_path:str='./experiment',
                       label:str='rcpyci',
                       n_jobs=10):
    
    # pass along input variables apart from already consumed ones
    kwargs = {k: v for k, v in locals().items()}
    consumed_variables = ['conditions', 'n_jobs']
    for key in consumed_variables:
        kwargs.pop(key, None)
    
    logging.info("Started calculating ci and zmaps for conditions. "
                    "Be mindful that with higher parallel jobs the progress bar becomes more inaccurate. "
                    "This may take a while... ")
    result = Parallel(n_jobs=n_jobs)(delayed(process_condition)(
        condition=condition,
        **kwargs
    ) for condition in tqdm(conditions))

    logging.info("Finished processing ci and zmaps for conditions.")
    return result

@skip_if_exist
def process_participant(participant,
                        data: pd.DataFrame,
                        base_image: np.ndarray,
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
                        seed: int = 1,
                        save_ci: bool = True,
                        save_zmap: bool = True,
                        experiment_path:str='./experiment',
                        label:str='rcpyci'):
    
    # pass along input variables apart from already consumed ones
    kwargs = {k: v for k, v in locals().items()}
    consumed_variables = ['participant', 'data', 'save_ci', 'save_zmap', 'experiment_path', 'label']
    for key in consumed_variables:
        kwargs.pop(key, None)
    
    # get normalized order of responses for the participant and pass to kwargs
    sorted_data = data[data['participant_id'] == participant].sort_values(by='stimulus_id')
    sorted_responses = sorted_data['responses'].to_numpy().reshape(n_trials, 1)
    kwargs['responses'] = sorted_responses

    ci, combined, zmap = compute_ci_and_zmap(**kwargs)
    
    ci_filename = f"ci_{label}_{participant}.png"
    if anti_ci:
        ci_filename = f"antici_{label}_{participant}.png"
    
    zmap_filename = f"zmap_{label}_{participant}.png"

    if save_ci:
        save_image(image=ci, path=os.path.join(experiment_path, "ci_raw", ci_filename), save_npy=True)
    if save_ci:
        save_image(image=combined, path=os.path.join(experiment_path, "ci", ci_filename), save_npy=True)
    if zmap_pipe is not None and save_zmap:
        save_image(image=zmap, path=os.path.join(experiment_path, "zmap", zmap_filename), save_npy=True)

    return participant, ci, combined, zmap

def process_participants(participants: list,
                         data: pd.DataFrame,
                         base_image: np.ndarray,
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
                         seed: int = 1,
                         save_ci: bool = True,
                         save_zmap: bool = True,
                         experiment_path:str='./experiment',
                         label: str='rcpyci',
                         n_jobs=10):
    """
    This function processes individual participants, calculating CI and zmaps for each.

    Parameters:
        participants (list): A list of participant IDs.
        data (pd.DataFrame): The raw response data.
        base_image (np.ndarray): The base image used for visualization.
        stimuli_params (np.ndarray, optional): Additional parameters related to the stimuli. Defaults to None.
        ci_postproc_pipe (Callable[[Any], Any], optional): A pipeline function for processing CI. Defaults to ci_postprocessing_pipeline.
        ci_postproc_kwargs (dict, optional): Keyword arguments for the CI post-processing pipeline. Defaults to default_ci_postprocessing_pipeline_kwargs.
        zmap_pipe (Callable[[Any], Any], optional): A pipeline function for computing zmaps. Defaults to compute_zmap_ttest_pipeline.
        zmap_kwargs (dict, optional): Keyword arguments for the zmap computation pipeline. Defaults to default_compute_zmap_ttest_pipeline_kwargs.
        anti_ci (bool, optional): Whether to generate anti-CI or not. Defaults to False.
        n_trials (int, optional): The number of trials used in the analysis. Defaults to 770.
        n_scales (int, optional): The number of scales used in the analysis. Defaults to 5.
        sigma (int, optional): The standard deviation used in the noise generation. Defaults to 5.
        noise_type (str, optional): The type of noise to generate. Defaults to 'sinusoid'.
        seed (int, optional): The random seed for generating noise. Defaults to 1.
        save_ci (bool, optional): Whether to save the calculated CI. Defaults to True.
        save_zmap (bool, optional): Whether to save the calculated zmaps. Defaults to True.
        experiment_path (str, optional): The path where the results will be saved. Defaults to './experiment'.
        label (str, optional): A unique label for the experiment. Defaults to 'rcpyci'.

    Returns:
        result (list): A list of participant IDs with corresponding CI and zmap data.

    """
    # pass along input variables apart from already consumed ones
    kwargs = {k: v for k, v in locals().items()}
    consumed_variables = ['participants', 'n_jobs']
    for key in consumed_variables:
        kwargs.pop(key, None)

    logging.info("Started calculating ci and zmaps for individual participants.  " +
                  "Be mindful that with higher parallel jobs the progress bar becomes more inaccurate.  " +
                    "This may take a while... ")
    result = Parallel(n_jobs=n_jobs)(delayed(process_participant)(participant=participant, **kwargs) for participant in tqdm(participants))
    logging.info("Finished processing individual ci and zmaps.")
    return result

def analyze_data(data: pd.DataFrame,
                 base_face_path:str,
                 stimuli_params: np.ndarray = None,
                 ci_postproc_pipe=ci_postprocessing_pipeline,
                 ci_postproc_kwargs=ci_postprocessing_pipeline_kwargs,
                 zmap_pipe=compute_zmap_ttest_pipeline,
                 zmap_kwargs=compute_zmap_ttest_pipeline_kwargs,
                 anti_ci=False,
                 n_trials: int = 770,
                 n_scales: int = 5,
                 sigma: int = 5,
                 noise_type: str = 'sinusoid',
                 seed: int = 1,
                 save_ci: bool = True,
                 save_zmap: bool = True,
                 experiment_path:str='./experiment',
                 label:str='rcpyci',
                 n_jobs=10):
    """
    This function analyzes the given data, a pandas DataFrame, to calculate classification images (CIs) for each participant based on their responses.
    It also computes z-maps if the necessary argument is provided.

    Parameters:
    - `data`: The input data, a pandas DataFrame. It must contain at least two columns: 'participant_id' and 'responses'.
    - `base_face_path`: The path to the base face image used in the experiment.
    - `stimuli_params`: Optional parameters for the stimuli used in the experiment (default is None).
    - `ci_postproc_pipe`: A pipeline function for post-processing CI calculations (default is ci_ postprocessing_pipeline).
    - `ci_postproc_kwargs`: Keyword arguments for the CI post-processing pipeline (default is default_ci_postprocessing_pipeline_ kwargs).
    - `zmap_pipe`: A pipeline function for computing z-maps (default is compute_zmap_ttest_pipeline).
    - `zmap_kwargs`: Keyword arguments for the z-map computation pipeline (default is default_compute_zmap_ttest_pipeline_ kwargs).
    - `anti_ci`: A boolean flag indicating whether to calculate anti-CI or not (default is False).
    - `n_trials`: The number of trials in the experiment (default is 770).
    - `n_scales`: The number of scales used in the experiment (default is 5).
    - `sigma`: The standard deviation of the noise added to the responses (default is 5).
    - `noise_type`: The type of noise to add to the responses (default is 'sinusoid').
    - `seed`: A seed value for reproducibility (default is 1).
    - `save_ci`: A boolean flag indicating whether to save the CI images or not (default is True).
    - `save_zmap`: A boolean flag indicating whether to save the z-map images or not (default is True).
    - `experiment_path`: The path where the experiment results will be saved (default is './experiment').
    - `label`: A label for the experiment results (default is 'rcpyci').
    - `n_jobs`: The number of parallel jobs to use for processing participants (default is 10).

    Returns:
    - A tuple containing two lists: `participants_results` and `conditions_results`. These lists contain the CI and z-map results for each participant and condition, respectively.

    Note that this function uses the multiprocessing library to process the data in parallel. The number of parallel jobs can be controlled using the `n_jobs` parameter.
    """
    base_image = read_image(os.path.join(os.getcwd(), base_face_path), grayscale=True)
    os.makedirs(experiment_path, exist_ok=True)

    # pass along input variables apart from already consumed ones
    kwargs = {k: v for k, v in locals().items()}
    consumed_variables = ['base_face_path']
    for key in consumed_variables:
        kwargs.pop(key, None)
    
    participants = list(data['participant_id'].unique())
    kwargs['participants'] = participants
    participants_results = process_participants(**kwargs)
    kwargs.pop('participants', None)

    conditions = list(data['condition'].unique())
    kwargs['conditions'] = conditions
    conditions_results = process_conditions(**kwargs)
    
    return participants_results, conditions_results


def setup_experiment(base_face_path: str,
                     n_trials: int = 770,
                     n_scales: int = 5,
                     sigma: int = 5,
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
        sigma (int): Sigma value used for Gaussian noise generation. Default
            is 5.
        noise_type (str): Type of noise to generate. Default is 'sinusoid'.
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
    stimuli_ori, stimuli_inv = generate_stimuli_2IFC(base_face=base_image,
                                                     n_trials=n_trials,
                                                     n_scales=n_scales,
                                                     sigma=sigma,
                                                     noise_type=noise_type,
                                                     seed=seed)
    
    logging.info("Creating folders and saving data to disk")
    os.makedirs(os.path(experiment_path), exist_ok=True)
    timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M")
    for trial, (stimulus, stimulus_inverted) in tqdm(enumerate(zip(stimuli_ori, stimuli_inv)), desc="Processing", total=len(stimuli_ori)):
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
             sigma=sigma,
             noise_type=noise_type,
             experiment_path=experiment_path,
             label=label,
             seed=seed)
    logging.info("Done!")

##### 
# setup_experiment("./base_face.jpg")
#####
# a = create_test_data()
# verify_data(a)
# results = analyze_data(a, "./base_face.jpg")
#####


a = create_test_data(n_trials=500)
a = pd.read_csv('mturk_data_for_rcpyci.csv')
verify_data(a)
stims = np.load('stimulus.npy')
results = analyze_data(a, "./base_face.jpg", n_trials=500, stimuli_params=stims)
a = 0
