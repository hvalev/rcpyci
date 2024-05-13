import numpy as np
import math
import sys
# Add python library path so we can import python functions 
# without a package for the reference implementation of rcicr
sys.path.append("/tests/pyrcicr_ref/pyrcicr_ref/")

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from generate_noise import generate_sinusoid, generate_gabor, generate_noise_pattern, generate_scales
from generateNoiseImage import generate_noise_image
from generateStimuli2IFC import generate_stimuli_2IFC

# Load the R package
package_name = "rcicr"
package = importr(package_name)

# Function to compare arrays
def assert_array_equal(arr1, arr2):
    assert np.array_equal(arr1, arr2), "Arrays are not equal"

def assert_array_close(arr1, arr2, rtol=1e-5, atol=1e-8):
    assert np.allclose(arr1, arr2, rtol=rtol, atol=atol), "Arrays are not close"

# Call an R function from the package
def call_r_function(func_name, input_params, python_func=None, assert_func=assert_array_equal):
    result_r = getattr(package, func_name)(*input_params)
    result_r = np.array(robjects.conversion.rpy2py(result_r))
    if python_func:
        result_python = python_func(*input_params)
        assert_func(result_r, result_python)
    return result_r

# Test generateSinusoid function
sinusoid_input_params = (512, 2, 90, math.pi/2, 1.0)
call_r_function("generateSinusoid", sinusoid_input_params, generate_sinusoid)
print("Generated sinusoid noise identical")

# Test generateGabor function
gabor_input_params = (512, 2, 90, math.pi/2, 25, 1.0)
call_r_function("generateGabor", gabor_input_params, generate_gabor, assert_array_close)
print("Generated gabor noise allclose")

# Test generateScales function
generate_scales_input_params = (512, 5)
call_r_function("generateScales", generate_scales_input_params, generate_scales)
print("Generated identical scales")

# Test generateNoisePattern function with sinusoid noise
generate_noise_pattern_input_params_1 = (512, 5, 'sinusoid', 25, False)
result_noise_pattern_r_unconv_sinusoid = package.generateNoisePattern(*generate_noise_pattern_input_params_1)

# Convert the result to a Python dictionary
result_noise_pattern_r = robjects.conversion.rpy2py(result_noise_pattern_r_unconv_sinusoid)
result_dict = {
    'patches': np.array(result_noise_pattern_r.rx2('patches')),
    'patchIdx': np.array(result_noise_pattern_r.rx2('patchIdx')),
    'noise_type': str(result_noise_pattern_r.rx2('noise_type')[0])
}

result_noise_pattern_python_sinusoid = generate_noise_pattern(*generate_noise_pattern_input_params_1)

# Assertions for each element
assert_array_equal(result_dict['patches'], result_noise_pattern_python_sinusoid['patches'])
print("Generated identical sinusoid noise patches")
assert_array_equal(result_dict['patchIdx'], result_noise_pattern_python_sinusoid['patchIdx'])
print("Generated identical sinusoid noise patch ids")
assert result_dict['noise_type'] == result_noise_pattern_python_sinusoid['noise_type']

# Test generateNoisePattern function with gabor noise
generate_noise_pattern_input_params_2 = (512, 5, 'gabor', 25, False)
result_noise_pattern_r_unconv_gabor = package.generateNoisePattern(*generate_noise_pattern_input_params_2)

# Convert the result to a Python dictionary
result_noise_pattern_r = robjects.conversion.rpy2py(result_noise_pattern_r_unconv_gabor)
result_dict = {
    'patches': np.array(result_noise_pattern_r.rx2('patches')),
    'patchIdx': np.array(result_noise_pattern_r.rx2('patchIdx')),
    'noise_type': str(result_noise_pattern_r.rx2('noise_type')[0])
}

result_noise_pattern_python_gabor = generate_noise_pattern(*generate_noise_pattern_input_params_2)

# Assertions for each element
assert_array_close(result_dict['patches'], result_noise_pattern_python_gabor['patches'])
difference = np.max(np.abs(result_dict['patches'] - result_noise_pattern_python_gabor['patches']))
print(f"Generated allclose gabor noise patches with max deviation of {difference}")

assert_array_close(result_dict['patchIdx'], result_noise_pattern_python_gabor['patchIdx'])
difference = np.max(np.abs(result_dict['patchIdx'] - result_noise_pattern_python_gabor['patchIdx']))
print(f"Generated allclose gabor noise patch ids with max deviation of {difference}")

assert result_dict['noise_type'] == result_noise_pattern_python_gabor['noise_type']

# Test generateNoiseImage function using a static params file and the previous identical sinusoid noise
# This effectively sideloads R's implementation equivalent of runif np.random.uniform array 
# to test the following functions
params = np.load('/tests/params.npy')
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate() # global conversion will be deprecated for local ones
# # Convert NumPy array to R np2rpy object
r_np2rpy_object = numpy2ri.numpy2rpy(params)
# copy of the arugments above
generate_noise_pattern_input_params_1 = (512, 5, 'sinusoid', 25, False)
result_noise_pattern_r_unconv_sinusoid = package.generateNoisePattern(*generate_noise_pattern_input_params_1)
result_noise_image_r = package.generateNoiseImage(r_np2rpy_object, result_noise_pattern_r_unconv_sinusoid)
numpy2ri.deactivate()
result_noise_pattern_python_sinusoid = generate_noise_pattern(*generate_noise_pattern_input_params_1)
result_noise_image_python = generate_noise_image(params, result_noise_pattern_python_sinusoid)

assert_array_close(result_noise_image_r, result_noise_image_python)
difference = np.max(np.abs(result_noise_image_r - result_noise_image_python))
print(f"Generated allclose noise images using sinusoid noise with max deviation of {difference}")


# Test generateStimuli2IFC function
import numpy as np
from rpy2.robjects import numpy2ri
import rpy2.robjects as ro
numpy2ri.activate() # global conversion will be deprecated for local ones
n_trials = 5
nscales = 5
nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
params = np.random.uniform(-1, 1, size=(n_trials, nparams))
r_np2rpy_params = numpy2ri.numpy2rpy(params)
base_face_files_dict = {'aName': 'base_face.jpg'}
base_face_files = ro.ListVector(base_face_files_dict)
result_r_unconv = package.generateStimuli2IFC(base_face_files=base_face_files,
                                              overwrite_params=params, 
                                              n_trials=n_trials, 
                                              ncores=4, 
                                              return_as_dataframe=True)
result_r = np.array(robjects.conversion.rpy2py(result_r_unconv))

result_python_unconv = generate_stimuli_2IFC(base_face_files=base_face_files_dict,
                                      overwrite_params=True, 
                                      params_values=params,
                                      n_trials=n_trials, 
                                      img_size=512, 
                                      noise_type='sinusoid', 
                                      nscales=nscales, 
                                      sigma=25,
                                      return_as_dataframe=True)
result_p = result_python_unconv.reshape(result_r.shape, order='F')

assert_array_close(result_r, result_p)
difference = np.max(np.abs(result_r - result_p))
print(f"Generated allclose stimuli images with max deviation of {difference}")

print("--------------- generate tests completed ---------------- ")