import sys
import os
import numpy as np
sys.path.append("/tests/pyrcicr_ref/pyrcicr_ref/")

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from generateCI import generate_CI, process_quick_zmap, process_ttest_zmap

package_name = "rcicr"
package = importr(package_name)

# little snippet to detect the created rdata/npz files needed to test the ci functions
import glob
def find_file(folder_path, extention):
    # Use glob to find all files with .RData extension in the specified folder
    return glob.glob(os.path.join(folder_path, extention))

# Example usage
folder_path = "/tests/stimuli"
rdata_file = find_file(folder_path, "*.Rdata")[0]
npz_file = find_file(folder_path, "*.npz")[0]
print(f"Found rdatafile @ {folder_path} named {rdata_file}")
print(f"Found npzfile @ {folder_path} named {npz_file}")

#TODO: what are we going to test here
#1. normal CIs
#2. inverted (anti-) CIs
#3. generating CIs for a single participant
#4. generating CIs for multiple participants (aggregated)
#5. zmap quick
#6. zmap t.test
#7. zmap quick (aggregated for multiple participants)
#8. zmap t.test (aggregated for multiple participants)

#https://github.com/zllrunning/face-parsing.PyTorch -> for creating mask
a = robjects.IntVector([1,2,3,4,5])
b = robjects.IntVector([1,1,1,1,1])
result = package.generateCI(stimuli=a,
                            responses=b, 
                            baseimage="aName",
                            antiCI=False,
                            sigma=3,
                            zmap=True,
                            zmapmethod='quick',
                            threshold=1,
                            rdata=rdata_file,
                            rdatapath=True)

# This is what the original implementation returns:
# return(list(ci=ci, scaled=scaled, base=base, combined=combined, zmap=zmap))
# Since it is a list, when we convert to numpy below we get a (5 X img_size X img_size) array
# where the first dimension contains each of those 5 elements: ci, scaled, base, combined, zmap
# This allows us directly to compare arrays without saving to and loading from files 
result_r = np.array(robjects.conversion.rpy2py(result))

# Convert the result to a Python object if necessary
result_python = generate_CI(stimuli=[0,1,2,3,4],
                            responses=[1,1,1,1,1],
                            baseimage="aName", 
                            anti_CI=False,
                            sigma=3,
                            zmap=True,
                            zmapmethod='quick',
                            threshold=1,
                            rdata=npz_file)

assert np.allclose(result_r[0,:,:], result_python['ci'])
if np.allclose(result_r[0,:,:], result_python['ci']):
    difference = np.max(np.abs(result_r[0,:,:] - result_python['ci']))
    print(f"generateCI 'ci' allclose output with max deviation of {difference}")

assert np.allclose(result_r[1,:,:], result_python['scaled'])
if np.allclose(result_r[1,:,:], result_python['scaled']):
    difference = np.max(np.abs(result_r[1,:,:] - result_python['scaled']))
    print(f"generateCI 'scaled' allclose output with max deviation of {difference}")

# another kind of assert. Basically that the smallest element-wise difference is below 0.04
# there is definitely some rounding errors propagating... but the end result is the same
assert abs(result_r[2,:,:] - result_python['base']).max() < 0.04
if abs(result_r[2,:,:] - result_python['base']).max() < 0.04:
    difference = np.max(np.abs(result_r[2,:,:] - result_python['base']))
    print(f"generateCI 'base' allclose output with max deviation of {difference}")

# another kind of assert. Basically that the smallest element-wise difference is below 0.04
# there is definitely some rounding errors propagating... but the end result is the same
assert abs(result_r[3,:,:] - result_python['combined']).max() < 0.04
if abs(result_r[3,:,:] - result_python['combined']).max() < 0.04:
    difference = np.max(np.abs(result_r[3,:,:] - result_python['combined']))
    print(f"generateCI 'combined' allclose output with max deviation of {difference}")


######## Test zmap functions in separation ##########
import numpy as np
from rpy2.robjects import numpy2ri
img_size = 512
sigma = 3
threshold = 1
numpy2ri.activate() # global conversion will be deprecated for local ones
# Convert NumPy array to R np2rpy object
# init stuff for generating the params stuff
# regenerate the params or use a pre-computed and saved value
test_ci = np.random.uniform(0, 1, size=(512,512))

#test_ci = np.arange(50, step=2).reshape((5,5))
r_np2rpy_test_ci = numpy2ri.numpy2rpy(test_ci)

n_trials = 5
nscales = 5
nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
params = np.random.uniform(-1, 1, size=(n_trials, nparams))
r_np2rpy_params = numpy2ri.numpy2rpy(params)

#noise_image_r = package.generateNoiseImage(r_np2rpy_object, result_noise_pattern_r_unconv)
# deactivate the automatic covnersions
numpy2ri.deactivate()


result = package.process_quick_zmap_step_by_step(ci=r_np2rpy_test_ci,
                                    sigma=sigma, 
                                    threshold=threshold,
                                    img_size=img_size)
result_r = np.array(robjects.conversion.rpy2py(result))

# this is to test whether the as image R function introduces some noise. It doesn't.
# The blurring operation is implemented differently with some changes to 
# 1) how the gaussian width is calculated and subsequently affecting pixel cells and
# 2) the padding added to the borders of the image which influences how much the values decrease within the image
# z = package.process_quick_zmap_step_by_step_asim(ci=r_np2rpy_test_ci)
# result_z = np.array(robjects.conversion.rpy2py(z))

result_python = process_quick_zmap(ci=test_ci,
                                   sigma=sigma,
                                   threshold=threshold)

difference = np.max(np.abs(result_r[0,:,:] - result_python['blurred_ci']))
print(f"zmap using 'quick' method for 'blurred_ci' results with max deviation of {difference}")
difference = np.max(np.abs(result_r[1,:,:] - result_python['scaled_image']))
print(f"zmap using 'quick' method for 'scaled_image' results with max deviation of {difference}")
with np.errstate(invalid='ignore'):
    difference = np.nanmax(np.abs(result_r[2,:,:] - result_python['zmap']))
    print(f"zmap using 'quick' method for 'zmap' results with max deviation of {difference}")

num_nans_r = np.sum(np.isnan(result_r[2,:,:]))
num_nans_python = np.sum(np.isnan(result_python['zmap']))
print("zmap using 'quick' method for 'zmap' results with nan counts "
      f"(i.e. values outside of threshold range of value {threshold}) with " 
      f"{num_nans_r} nans in the result of the r implementation and {num_nans_python} -- for the python one")


##################
# the p argument (patchIdx and patches) is derived from the rdata file generated upon generating stimuli
# that's why we are using for the test the sideloaded function, which is a carbon copy of the original
# but sideloading the patches and patch indices
result = package.process_ttest_zmap_sideloaded(params=r_np2rpy_params,
                                    responses=b,
                                    img_size=img_size,
                                    ci=r_np2rpy_test_ci,
                                    rdata=rdata_file, 
                                    rdatapath=True)

result_r = np.array(robjects.conversion.rpy2py(result))

result_python = process_ttest_zmap(params=params,
                                   responses=np.array([1,1,1,1,1]),
                                   img_size=img_size, 
                                   ci=test_ci,
                                   rdata=npz_file)

assert np.allclose(result_r, result_python), "Arrays are not equal"
difference = np.max(np.abs(result_r - result_python))
print("\n")
print(f"zmap using 't.test' method results in allclose output with max deviation of {difference}")

print("--------------- ci tests completed ---------------- ")