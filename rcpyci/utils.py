"""
File Name: core.py
Description: object to object part of the library, for any interaction and interfacing with files, check interface.py
"""
import os
import random
from functools import wraps

import numpy as np
import pandas as pd
from PIL import Image


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
        if kwargs['save_zmap'] and not os.path.exists(os.path.join(kwargs['experiment_path'], "zmap", zmap_filename)):
            zmap_exists = False
        
        # If both files exist, skip computation
        if ci_exists and zmap_exists:
            print(f"Skipping computation for participant/condition {nameplate}: ci and zmap files already exist.")
            return nameplate, None, None, None
        
        # Otherwise, proceed with the original function
        return func(*args, **kwargs)
    
    return wrapper


def create_test_data(n_participants:int=100, n_trials:int=770):
    # for sample data reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    conditions = ["cond1", "cond2", "cond3", "cond4", "cond5"]
    data = {
        'idx': np.arange(0, n_participants*n_trials),
        'condition': np.concatenate([np.tile(np.random.choice(conditions), n_trials) for _ in range(n_participants)]),
        'participant_id': np.repeat(np.arange(0, n_participants), n_trials),
        'stimulus_id': np.tile(np.arange(0, n_trials), n_participants),
        'responses': np.random.choice([-1, 1], size=n_participants*n_trials).astype(int),
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

    # Step 5: Assess response distribution
    vprint("\nDistribution of responses:", verbose)
    vprint(df['responses'].value_counts(), verbose)
    responses = df['responses'].unique()
    if not (len(responses) == 2 and -1 in responses and 1 in responses):
        print('ERROR: Found a value in responses different from 1 or -1')

# def compare_images(img1_path, img2_path):
#     return all([x == y for x, y in zip(Image.open(img1_path).convert("L").getdata(), Image.open(img2_path).convert("L").getdata())])


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
