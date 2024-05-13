import sys
sys.path.append("/tests/pyrcicr_ref/pyrcicr_ref")
#import os
#os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+':/usr/local/lib/R/lib:/usr/local/lib/R/modules'
#os.environ['R_HOME'] = '/usr/local/lib/R'
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import math
from generate_noise import generate_sinusoid, generate_gabor, generate_noise_pattern, generate_scales
from generateNoiseImage import generate_noise_image

#from generateStimuli2IFC import generateStimuli2IFC
from generateStimuli2IFC import generate_stimuli_2IFC
#from generate_noise import generate_gabor, generate_sinusoid, generate_noise_pattern
# import os
# os.environ['R_HOME'] = '/usr/local/lib/R'

#TODO compare arrays as following:
# 1. equivalent
# 2. all close -> define some metrics (like e-16/e-17/e-18 to guarantee almost sameness)
# 3. give some feedback on how close the arrays are, some margin of error (max/min/median over the arrays element-wise) biggest/smallest difference

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/R/lib:/usr/local/lib/R/modules
# Load the R package
package_name = "rcicr"
package = importr(package_name)


# Call an R function from the package
# img_size, cycles, angle, phase, contrast
sinusoid_input_params = (512, 2, 90, math.pi/2, 1.0)
result = package.generateSinusoid(*sinusoid_input_params)

# Convert the result to a Python object if necessary
result_r = np.array(robjects.conversion.rpy2py(result))
result_python = generate_sinusoid(*sinusoid_input_params)

assert np.array_equal(result_r, result_python)
if np.array_equal(result_r, result_python):
    print("generate_sinusoid results in equal output between R and Python")

# img_size, cycles, angle, phase, sigma, contrast
gabor_input_params = (512, 2, 90, math.pi/2, 25, 1.0)
result = package.generateGabor(*gabor_input_params)

# Convert the result to a Python object if necessary
result_r = np.array(robjects.conversion.rpy2py(result))
result_python = generate_gabor(*gabor_input_params)

assert np.allclose(result_r, result_python)
if np.allclose(result_r, result_python):
    print("generate_gabor results in allclose output between R and Python")


# Call an R function from the package
# img_size, cycles, angle, phase, contrast
generate_scales_input_params = (512, 5)
result = package.generateScales(*generate_scales_input_params)

# Convert the result to a Python object if necessary
result_r = np.array(robjects.conversion.rpy2py(result))
result_python = generate_scales(*generate_scales_input_params)

assert np.array_equal(result_r, result_python)
if np.array_equal(result_r, result_python):
    print("generate_scales results in equal output between R and Python")


# img_size, cycles, angle, phase, sigma, contrast
#img_size=512, nscales=5, noise_type='sinusoid', sigma=25, pre_0.3.0=FALSE
generate_noise_pattern_input_params_1 = (512, 5, 'sinusoid', 25, False)
result_noise_pattern_r_unconv = package.generateNoisePattern(*generate_noise_pattern_input_params_1)

# Convert the result to a Python object if necessary
result_noise_pattern_r = robjects.conversion.rpy2py(result_noise_pattern_r_unconv)
result_noise_pattern_r = dict(zip(result_noise_pattern_r.names, result_noise_pattern_r))
result_noise_pattern_r['patches'] = np.array(result_noise_pattern_r['patches'])
result_noise_pattern_r['patchIdx'] = np.array(result_noise_pattern_r['patchIdx'])
result_noise_pattern_r['noise_type'] = str(result_noise_pattern_r['noise_type'])
result_noise_pattern_python = generate_noise_pattern(*generate_noise_pattern_input_params_1)

assert np.array_equal(result_noise_pattern_r['patches'], result_noise_pattern_python['patches'])
if np.array_equal(result_noise_pattern_r['patches'], result_noise_pattern_python['patches']):
    print("generate_noise_pattern patches results in array_equal output between R and Python using sinusoid")
assert np.array_equal(result_noise_pattern_r['patchIdx'], result_noise_pattern_python['patchIdx'])
if np.array_equal(result_noise_pattern_r['patchIdx'], result_noise_pattern_python['patchIdx']):
    print("generate_noise_pattern patchIdx results in array_equal output between R and Python using sinusoid")

#img_size=512, nscales=5, noise_type='sinusoid', sigma=25, pre_0.3.0=FALSE
generate_noise_pattern_input_params_2 = (512, 5, 'gabor', 25, False)
result = package.generateNoisePattern(*generate_noise_pattern_input_params_2)

# Convert the result to a Python object if necessary
result_r = robjects.conversion.rpy2py(result)
result_r = dict(zip(result_r.names, result_r))
result_r['patches'] = np.array(result_r['patches'])
result_r['patchIdx'] = np.array(result_r['patchIdx'])
result_r['noise_type'] = str(result_r['noise_type'])
result_python = generate_noise_pattern(*generate_noise_pattern_input_params_2)

assert np.allclose(result_r['patches'], result_python['patches'])
if np.allclose(result_r['patches'], result_python['patches']):
    print("generate_noise_pattern patches results in allclose output between R and Python using gabor")
assert np.allclose(result_r['patchIdx'], result_python['patchIdx'])
if np.allclose(result_r['patchIdx'], result_python['patchIdx']):
    print("generate_noise_pattern patchIdx results in allclose output between R and Python using gabor")


# result = package.generateNoiseImageMinRepro(n_trials=770, img_size=512, noise_type='sinusoid', nscales=5, sigma=25, ncores=3, return_as_dataframe=True)
# load some precomputed params. We know it's the same as the R equivalent
# because we know the sinusoid noise is equivalent
params = np.load('/pyrcicr/tests/params.npy')
# important to be able to convert numpy arrays to R
# https://rpy2.github.io/doc/latest/html/numpy.html
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate() # global conversion will be deprecated for local ones
# Convert NumPy array to R np2rpy object
r_np2rpy_object = numpy2ri.numpy2rpy(params)
noise_image_r = package.generateNoiseImage(r_np2rpy_object, result_noise_pattern_r_unconv)
# deactivate the automatic covnersions
numpy2ri.deactivate()
noise_image_python = generate_noise_image(params, result_noise_pattern_python)
assert np.allclose(noise_image_r, noise_image_python)
if np.allclose(noise_image_r, noise_image_python):
    print("generate_noise_image results in allclose output between R and Python using sinusoid")


# Call an R function from the package
#TODO make the input variables into keyword arguments
#generate_stimulus_input_param = (base_face_files, n_trials=770, img_size=512, stimulus_path='./stimuli', label='rcic', use_same_parameters=True, seed=1, maximize_baseimage_contrast=True, noise_type='sinusoid', nscales=5, sigma=25, ncores=-1, return_as_dataframe=False, save_as_png=True, save_rdata=True)
generate_stimulus_input_param = {
    'base_face_files': {'aName': "base_face.jpg"},
    'n_trials': 770,
    'img_size': 512,
    'stimulus_path': './stimuli',
    'label': 'rcic',
    'use_same_parameters': True,
    'seed': 1,
    'maximize_baseimage_contrast': True,
    'noise_type': 'sinusoid',
    'nscales': 5,
    'sigma': 25,
    'ncores': -1,
    'return_as_dataframe': True,
    'save_as_png': True,
    'save_rdata': True
}
kwargs = list(generate_stimulus_input_param.values())
base = list('aNames=base_face.jpg')

# # Importing the required module
# import rpy2.robjects as ro

# # Creating a dictionary with key-value pairs
# data = {"aName": "/pyrcicr/R/www/base_face.jpg"}



# # Converting the dictionary to an R list
# r_list = ro.ListVector(data)

# # Printing the R list
# print(r_list)

import rpy2.robjects as ro

# Create a named list in R
base_face_files = ro.ListVector({'aName': 'base_face.jpg'})

# Now you can pass this list to an R function
# For demonstration, let's print the structure of this list in R
ro.r['print'](base_face_files)


#TODO sideload runif np.random.uniform array to test function
#This part of the code substitutes the randomly generated array with one generated by python instead
#as we cannot generate procedurally the same numbers between python and R even with setting the same seed
#as a sequence the R function has been modified slightly to accept an external array as parameters when provided
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate() # global conversion will be deprecated for local ones
# Convert NumPy array to R np2rpy object
# init stuff for generating the params stuff
# regenerate the params or use a pre-computed and saved value
n_trials = 5
nscales = 5
nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
params = np.random.uniform(-1, 1, size=(n_trials, nparams))
r_np2rpy_params = numpy2ri.numpy2rpy(params)
#noise_image_r = package.generateNoiseImage(r_np2rpy_object, result_noise_pattern_r_unconv)
# deactivate the automatic covnersions
numpy2ri.deactivate()
file_path = '/pyrcicr/tests/test.npy'
#if not os.path.exists(file_path):
result = package.generateStimuli2IFC(base_face_files=base_face_files, overwrite_params=r_np2rpy_params, n_trials=n_trials, ncores=4, return_as_dataframe=True)
# Convert the result to a Python object if necessary
# (770, 262144)
result_r = np.array(robjects.conversion.rpy2py(result))
#     with open(file_path, 'wb') as f:
#         np.save(f, result_r)
# else:
#     result_r = np.load(file_path)


#result_python = generateStimuli2IFC(**generate_stimulus_input_param)python3 -c "import cv2; print(cv2.__version__)"
#result_python = generateStimuli2IFC(*kwargs)
result_python = generate_stimuli_2IFC(base_face_files={'aName': 'base_face.jpg'}, overwrite_params=True, params_values=params,
                                    n_trials=n_trials, img_size=512, noise_type='sinusoid', nscales=nscales, sigma=25,
                                    return_as_dataframe=True)
#R reshapes array in (column-major) fortran style, while python does it in row-major (c-style)
#thus we need to reshape either the R's or python's array (here python's) to ensure we are comparing
#the results correctly element-wise
result_p = result_python.reshape(result_r.shape, order='F')

# generated images look the same, but the results aren't
assert np.allclose(result_r, result_p)
if np.allclose(result_r, result_p):
    print("generatestimuli2ifc results in allclose output between R and Python")

