import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def batchGenerateCI2IFC(data, by, stimuli, responses, baseimage, rdata, save_as_png=True, targetpath='./cis', label='', antiCI=False, scaling='autoscale', constant=0.1):
    if scaling == 'autoscale':
        do_autoscale = True
        scaling = 'none'
    else:
        do_autoscale = False

    cis = {}

    # Remove rows with NA in the 'by' column
    data = data.dropna(subset=[by])

    by_levels = data[by].unique()
    pb = tqdm(total=len(by_levels), unit=' unit')

    for unit in by_levels:
        # Update progress bar
        pb.update(1)

        # Get subset of data
        unitdata = data[data[by] == unit]

        # Specify filename for CI PNG
        if not label:
            filename = f"{baseimage}_{by}_{unitdata.iloc[0][by]}"
        else:
            filename = f"{baseimage}_{label}_{by}_{unitdata.iloc[0][by]}"

        # Compute CI with appropriate settings for this subset (Optimize later so rdata file is loaded only once)
        cis[filename] = generateCI2IFC(
            stimuli=unitdata[stimuli].to_numpy(),
            responses=unitdata[responses].to_numpy(),
            baseimage=baseimage,
            rdata=rdata,
            save_as_png=save_as_png,
            filename=filename,
            targetpath=targetpath,
            antiCI=antiCI,
            scaling=scaling,
            scaling_constant=constant,
        )

    if do_autoscale:
        cis = autoscale(cis, save_as_pngs=save_as_png, targetpath=targetpath)

    pb.close()
    return cis
