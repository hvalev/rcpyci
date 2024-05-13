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
from utils import create_test_data, verify_data, skip_if_exist

logging.basicConfig(level=logging.INFO)

def setup_experiment(base_face_path: str,
                    n_trials:int = 770,
                    n_scales:int=5,
                    sigma:int=5,
                    noise_type:str='sinusoid',
                    experiment_path:str='./experiment',
                    label:str='rcpyci',
                    seed:int=1):
    
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
