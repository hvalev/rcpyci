import numpy as np
from scipy.stats import norm, ttest_1samp
from skimage.filters import gaussian
from skimage.io import imsave


def generateCI2IFC(stimuli, responses, baseimage, rdata, save_as_png=True, filename='', targetpath='./cis', antiCI=False, scaling='independent', constant=0.1):

    # Wrap the generateCI function for backwards compatibility
    return generateCI(
        stimuli=stimuli,
        responses=responses,
        baseimage=baseimage,
        rdata=rdata,
        save_individual_cis=False,
        save_as_png=save_as_png,
        filename=filename,
        targetpath=targetpath,
        antiCI=antiCI,
        scaling=scaling,
        scaling_constant=constant,
        individual_scaling='independent',
        individual_scaling_constant=0.1,
        zmap=False,
        zmapmethod='quick',
        zmapdecoration=True,
        sigma=3,
        threshold=3,
        zmaptargetpath='./zmaps',
        n_cores=None,
        mask=None
    )
