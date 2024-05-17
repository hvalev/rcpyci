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
from core import compute_ci_and_zmap, generate_stimuli_2IFC
from im_ops import read_image, save_image
from joblib import Parallel, delayed
from pipelines import (
    ci_postprocessing_pipeline,
    compute_zmap_ttest_pipeline,
    default_ci_postprocessing_pipeline_kwargs,
    default_compute_zmap_ttest_pipeline_kwargs,
)
from tqdm import tqdm
from utils import create_test_data, skip_if_exist, verify_data

logging.basicConfig(level=logging.INFO)

@skip_if_exist
def process_condition(condition,
                      data: pd.DataFrame,
                      base_image: np.ndarray,
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
        save_image(image=combined, path=os.path.join(experiment_path, "ci", ci_filename), save_npy=True)
    if zmap is not None and save_zmap:
        save_image(image=zmap, path=os.path.join(experiment_path, "zmap", zmap_filename), save_npy=True)

    return condition, ci, combined, zmap

def process_conditions(conditions, 
                       data: pd.DataFrame,
                       base_image: np.ndarray,
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
                        ci_postproc_kwargs: dict = default_ci_postprocessing_pipeline_kwargs,
                        zmap_pipe: Callable[[Any], Any] = compute_zmap_ttest_pipeline,
                        zmap_kwargs: dict = default_compute_zmap_ttest_pipeline_kwargs,
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
        save_image(image=combined, path=os.path.join(experiment_path, "ci", ci_filename), save_npy=True)
    if zmap is not None and save_zmap:
        save_image(image=zmap, path=os.path.join(experiment_path, "zmap", zmap_filename), save_npy=True)

    return participant, ci, combined, zmap

def process_participants(participants: list,
                         data: pd.DataFrame,
                         base_image: np.ndarray,
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
                         seed: int = 1,
                         save_ci: bool = True,
                         save_zmap: bool = True,
                         experiment_path:str='./experiment',
                         label: str='rcpyci',
                         n_jobs=10):
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
                 ci_postproc_kwargs=default_ci_postprocessing_pipeline_kwargs,
                 zmap_pipe=compute_zmap_ttest_pipeline,
                 zmap_kwargs=default_compute_zmap_ttest_pipeline_kwargs,
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
