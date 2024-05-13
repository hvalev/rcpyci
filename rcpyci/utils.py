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
