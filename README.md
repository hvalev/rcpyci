# rcpyci 🌶️
[![build](https://github.com/hvalev/rcpyci/actions/workflows/build.yml/badge.svg)](https://github.com/hvalev/rcpyci/actions/workflows/build.yml)
[![Downloads](https://static.pepy.tech/badge/rcpyci)](https://pepy.tech/project/rcpyci)
[![Downloads](https://static.pepy.tech/badge/rcpyci/month)](https://pepy.tech/project/rcpyci)
[![Downloads](https://static.pepy.tech/badge/rcpyci/week)](https://pepy.tech/project/rcpyci)
[![DOI](https://zenodo.org/badge/795065661.svg)](https://doi.org/10.5281/zenodo.15827936)

rcpyci (/ɑr ˈspaɪsi/) is a reverse correlation classification images toolbox for generating classification images for 2AFC experiments and analyzing experiment results. rcpyci supports the following tasks:
- generate stimulus images for 2AFC tasks
- create classification images from 2AFC tasks split by participant or condition
- compute zmaps on classification images or parameter space
- generic way to add additional pipelines for computing anything on the ci- or parameter space
- caching of intermediary results
- generating the parameter space using a seed

## How does this compare to the [rcicr](https://github.com/rdotsch/rcicr/) R package
The implementation of the core functions closely mirrors that of `rcicr`, but `rcpyci` is much faster, easy to use, and future-proof as it depends on a few regularly maintained python packages. Additionally, `rcpyci` exposes a way for the user to implement their own custom processing pipelines without needing to modify the source code. And it has caching and seeding.
#TODO

## What is in this package
The main components in this package are split in 5 namespaces:
- `core`: Functions for creating stimulus images and computing classification images using pure numpy arrays.
- `interface`: User-friendly interface for setting up experiments and analyzing data, wrapping the functionality of `core`.
- `pipelines`: A collection of functions for computing cis, zmaps, and others, executed in sequence on 2AFC task results. More on that in the [Default processing pipelines](#default-processing-pipelines) section.
- `im_ops`: Operations on image arrays.
- `utils`: Helper functions.

__NOTE__: `infoval` is a port of the `infoval` functionality from the original `rcicr` package, but it is untested.

## How to generate stimuli for a 2AFC task
Here is how you can generate 2AFC task stimuli:
```python
from rcpyci.interface import setup_experiment
base_face_path = "./base_face.jpg"
setup_experiment(base_face_path)
```
__NOTE__: The input face image needs to be square-shaped.

`setup_experiment` further exposes the parameters `n_trials`, `n_scales`, `gabor_sigma`, `noise_type` to control the noise generation and `seed` for reproducibility. Check the docstrings of `setup_experiment` for more information.

## How to analyze data from an experiment
The following snippet shows how experiment data can be analyzed. Here we are using some generated sample data.
```python
from rcpyci.interface import analyze_data
from rcpyci.utils import create_test_data, verify_data

sample_data = create_test_data(n_trials=500)
verify_data(sample_data)
base_face_path = "./base_face.jpg"
analyze_data(sample_data, base_face_path, n_trials=500)
```
__NOTE__: `analyze_data` takes a dataframe with 5 columns: `idx`, `condition`, `participant_id`, `stimulus_id` and `responses`. 
`idx` is the row identifier, `condition` is the condition that was used for each trial, (e.g. the question variation asked to a participant). If a single condition is tested, the column will have a single value. `participant_id` is an identifier for each participant. `stimulus_id` is an identifier for each stimulus (pair of original and inverted images generated by setup_experiment). `responses` is the response that was given by each participant for each stimulus image pair. If a participant has selected the original response image, then the value should be coded as `1`, if the inverted response image is selected -- as `-1`. There is no need to sort by stimulus ids as that is taken care of in the method.

__NOTE__: To speed-up computation, you can allocate multiple cores using `n_jobs`. It's a balancing game between number of cpu cores and available RAM. If you have around 20GB RAM, you could use around 6 concurrent jobs. The default parameter configuration uses caching, so you could stop and resume the computation at a later point without losing any progress.

## Default processing pipelines
The default processing pipelines are defined in `pipelines.full_pipeline`. A quick look: 
```
full_pipeline = [
    (compute_ci, compute_anti_ci_kwargs),
    (combine_ci, combine_anti_ci_kwargs),
    (compute_ci, compute_ci_kwargs),
    (combine_ci, combine_ci_kwargs),
    (compute_zmap_ci, compute_zmap_ci_kwargs),
    (compute_zmap_stimulus_params, compute_zmap_stimulus_params_kwargs)
]
```
Using the default pipelines, we compute in-order the ci, anti-ci, zmap on the ci and zmap on the parameter space for a given input, typically on a participant or a condition split with some sensible default parameters. This pipeline can be modified by adding or removing individual steps or modifying the `..._kwargs` parameters assigned to each of the steps.

### Custom processing pipelines
The pipelines can also be extended by adding your own functions with their respective parameters. Here is a simple example where we modify the seed and cache the modified seed as a numpy array in two steps.
```python
from rcpyci.interface import analyze_data
from rcpyci.utils import create_test_data, verify_data, cache_as_numpy

sample_data = create_test_data(n_trials=500)
verify_data(sample_data)
base_face_path = "./base_face.jpg"

sample_pipe_generator_kwargs = {
    'addition': 1000
}

def sample_pipe_generator(seed, addition):
    return {'modified_seed': seed + addition}

sample_pipe_receiver_kwargs = {
    'use_cache': True,
    'save_folder': 'sample'
}

@cache_as_numpy
def sample_pipe_receiver(modified_seed, cache_path=None):
    return {'modified_seed': modified_seed}

pipelines = [
    (sample_pipe_generator, sample_pipe_generator_kwargs),
    (sample_pipe_receiver, sample_pipe_receiver_kwargs)
]

analyze_data(sample_data, base_face_path, pipelines=pipelines, n_trials=500)
```
__NOTE__: If you want to leverage caching, use the `@cache_as_numpy` decorator and add `cache_path=None` to the function signature.

### Parameters always exposed to pipeline functions
By default, the following parameters can be used within all pipeline functions:
`base_image` - base image used in the experiment
`responses` - sorted responses to the stimulus images in the current split. E.g, 1 or -1 mapping the choice to original or inverted noise images 
`pipeline_id` - string combining the label and participant/codnition split. Used for caching.
`experiment_path`- relative path where data will be stored
`stimuli_params` - the parameter space used to generate the stimulus images
`patches` - noise patches used to generate the original and inverted stimulus images 
`patch_idx` - patch indices 
`n_trials` - number of trials
`n_scales` - number of scales used when generating the noise
`gabor_sigma` - sigma parameter when using gabor noise
`noise_type` - method used for generating noise. Either `sinusoid` or `gabor`
`seed` - seed value
To use either of them, just add the variable to the method signature. Be mindful that changing those values directly will persist for the rest of the pipeline run.

# Compatibility with R's rcicr
The implementation should produce the same results between this one and R's implementations with a few caveats. First, there are differences in how pythons' `numpy` and R's `random` packages generate random numbers. Even though `rcpyci` and `rcicr` can both be seeded, the output will differ. Second, there differences in how certain operations in the underlying libraries are implemented, which results in small numerical differences, the biggest being at computing the cis with roughly `~0.0005` difference. A comparison between the analogous functions can be ran using the `run_tests.sh` script from the `ref` folder in this repository. More info on that [here](ref/README.md).

## How to use rcicr stimuli in rcpyci
If you have experiment data created with the R `rcicr` package, you can still use `rcpyci` to process your data. As the random number generation approaches between python and R are not interchangeable, the parameter space generated by `rcicr` needs to be exported. After that, you can process your data `rcpyci`. 

The easiest way is to use the docker container which has a functional R environment as well as the necessary `rpy2` python package. It can be started as follows:
```bash
docker run -it -w / -v ./data:/data hvalev/rcpyci  /bin/bash
export R_HOME=/usr/local/lib/R && export LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib/R/
/pyrcicr/bin/python3
```
__NOTE__: This will create a data folder in the current working dir on your host, where the exported parameter space will be stored.

Make sure that you have your RData file created by `rcicr` in the `./data` folder. Afterwards you need to load and convert to to a numpy array like this:
```python
import rpy2.robjects as robjects
import numpy as np
robjects.r['load']("/data/test.RData")
z = np.array(robjects.conversion.rpy2py(robjects.r['stimuli_params']))
z.shape
```
In my case, the number of trials were 500, hence the shape of the reshaped array below. Feel free to adjust the number below to the number of trials used in your own experiment. Additionally, you can double-check that the max and min values in the converted array are within the (-1,1) interval as one would expect.
```python
t = z.reshape(500,4092)
np.save('/data/stimulus.npy', t)
t.max()
t.min()
```
With the stimulus.npy file in place you can analyze your data with `rcpyci` by loading and passing the numpy array as `stimulus_param` in the `analyze_data` function in `rcpyci.interface`. This would disable re-generating the parameter space using the provided seed value and use this sideloaded parameter space instead. You must make sure that you provide matching parameters for `n_scales` and `n_trials` so that the patches and patch indices are generated with the correct array dimensions.