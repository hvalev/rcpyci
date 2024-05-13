"""
File Name: core.py
Description: object to object part of the library, for any interaction and interfacing with files, check interface.py
"""
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from functools import wraps

def skip_if_exist(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # set both to true
        ci_exists = True
        zmap_exists = True

        nameplate = ''
        # are we computing participant-level cis and zmaps or condition-level cis
        if 'participant' in kwargs.keys():
            nameplate = kwargs['participant']
        elif 'condition' in kwargs.keys():
            nameplate = kwargs['condition']
        else:
            return func(*args, **kwargs)

        # Check if ci and zmap files exist
        ci_filename = f"ci_{kwargs['label']}_{nameplate}.png"
        if kwargs['anti_ci']:
            ci_filename = f"antici_{kwargs['label']}_{nameplate}.png"
        
        zmap_filename = f"zmap_{kwargs['label']}_{nameplate}.png"

        if kwargs['save_ci'] and not os.path.exists(os.path.join(kwargs['experiment_path'], "ci", ci_filename)):
            ci_exists = False
            #save_image(image=combined, path=os.path.join(experiment_path, "ci", ci_filename), save_npy=True)
        if kwargs['save_zmap'] and not os.path.exists(os.path.join(kwargs['experiment_path'], "zmap", zmap_filename)):
            zmap_exists = False
            #save_image(image=zmap, path=os.path.join(experiment_path, "zmap", zmap_filename), save_npy=True)
        
        # If both files exist, skip computation
        if ci_exists and zmap_exists:
            print(f"Skipping computation for participant/condition {nameplate}: ci and zmap files already exist.")
            return nameplate, None, None, None
        
        # Otherwise, proceed with the original function
        return func(*args, **kwargs)
    
    return wrapper

def create_test_data(num_participants:int=100, num_stimulus:int=770):
    # for sample data reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    conditions = ["cond1", "cond2", "cond3", "cond4", "cond5"]
    data = {
        'idx': np.arange(0, num_participants*num_stimulus),
        'condition': np.concatenate([np.tile(np.random.choice(conditions), num_stimulus) for _ in range(num_participants)]),
        'participant_id': np.repeat(np.arange(0, num_participants), num_stimulus),
        'stimulus_id': np.tile(np.arange(0, num_stimulus), num_participants),
        'stimulus_presentation_order': np.concatenate([np.random.permutation(num_stimulus) for _ in range(num_participants)]),
        'responses': np.random.choice([-1, 1], size=num_participants*num_stimulus).astype(int),
    }
    return pd.DataFrame(data)

def vprint(msg, verbose=False):
    if verbose:
        print(msg)

def verify_data(df: pd.DataFrame, verbose=False):
    # Step 1: Check dimensions
    vprint("Dimensions of DataFrame:", verbose)
    vprint(df.shape, verbose)

    # Report on missing values
    vprint("\nMissing values:", verbose)
    vprint(df.isnull().sum(), verbose)
    if len(df.isnull().sum().unique()) != 1:
        print("ERROR identified missing values. Please run validation with verbose=True")

    # Establish facts about the dataset
    vprint("\nDistribution of participant IDs:", verbose)
    vprint(df['participant_id'].value_counts(), verbose)
    print(f"Found {len(df['participant_id'].value_counts())} participants")
    if len(df['participant_id'].value_counts().unique()) == 1:
        n_trials = list(df['participant_id'].value_counts().unique())[0]
        print(f"Found {n_trials} trials per participants")
    
    # Step 3: Check distribution of conditions, participant IDs, and stimulus IDs
    vprint("\nDistribution of conditions:", verbose)
    vprint(df['condition'].value_counts(), verbose)
    if len(df['condition'].value_counts().unique()) != 1:
        print('WARNING conditions are not even between participants')
    

    vprint("\nDistribution of stimulus IDs:", verbose)
    vprint(df['stimulus_id'].value_counts(), verbose)
    if len(df['stimulus_id'].value_counts().unique()) != 1:
        print('ERROR not all stimuli have been presented evenly to participants')

    if 'stimulus_presentation_order' in df.columns:
        vprint("\nDistribution of stimulus IDs:", verbose)
        vprint(df['stimulus_presentation_order'].value_counts(), verbose)
        if len(df['stimulus_presentation_order'].value_counts().unique()) != 1:
            print('ERROR presentation order id not congruent with stimuli')

    # Step 5: Assess response distribution
    vprint("\nDistribution of responses:", verbose)
    vprint(df['responses'].value_counts(), verbose)
    responses = df['responses'].unique()
    if not (len(responses) == 2 and -1 in responses and 1 in responses):
        print('ERROR: Found a value in responses different from 1 or -1')

