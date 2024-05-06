import os
import random
from functools import partial
import numpy as np
from PIL import Image
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from datetime import datetime
import math
# Functions
from scipy.stats import norm, ttest_1samp
from scipy.ndimage import gaussian_filter

from .im_ops import save_image, read_image

#TODO better function names
#TODO fix stimuli and params to stimuli be indices and params the params (fixed already?)
#TODO clean up interfaces (filepath not needed)




# Main function
def generate_CI(stimuli, 
                responses, 
                img, 
                img_size=512, 
                noise_type='sinusoid', 
                nscales=5, 
                save_as_png=True, 
                label='experiment',
                targetpath='/home/hval/rcpyci/cis', 
                anti_CI=False, 
                scaling='independent',
                scaling_constant=0.1, 
                individual_scaling='independent',
                individual_scaling_constant=0.1, 
                zmap=False,
                zmapmethod='quick', 
                sigma=3,
                threshold=3, 
                zmaptargetpath='./zmaps',
                mask=None):
    # Load parameter file (created when generating stimuli)
    patches, patch_idx, noise_type = generate_noise_pattern(img_size=img_size, noise_type=noise_type, nscales=nscales, sigma=sigma)
    stimuli_params = generate_stimuli_params(n_trials, nscales)

    #TODO add variable participant (assuming already it's for one participant)
    #TODO when I add the parsing of the dataframe I can set it to condition
    #TODO OR I can have again a wrapper, which has participants and conditions 
    #TODO and this wrapper (similar to the R one) just formats the data and triggers this function to be executed
    filename = ''
    if anti_CI:
        filename += 'antici_' + label + ".png"
        stimuli_params = -stimuli_params
    else:
        filename += 'ci_' + label + ".png"

    # reorder based on selection order
    stimuli_params = stimuli_params[stimuli]
    ci = generate_ci_noise(stimuli_params, responses, patches, patch_idx)
    if mask is not None:
        ci = apply_mask(ci, mask)
    scaled = apply_scaling(img, ci, scaling, scaling_constant)
    combined = combine(scaled, img)

    if save_as_png:
        save_image(image=combined, path='/home/hval/rcpyci/cis/'+filename)

    # Z-map
    zmap_image = None
    if zmap:
        if zmapmethod == 'quick':
            blurred_ci, scaled_image, zmap_image = process_quick_zmap(ci, sigma, threshold)
        elif zmapmethod == 't.test':
            zmap_image = process_ttest_zmap(stimuli_params, responses, patches, patch_idx, img_size, ci)
            np.save("/home/hval/rcpyci/zmap/zmap.npy", zmap_image)
        else:
            raise ValueError(f"Invalid zmap method: {zmapmethod}")
    return ci, scaled, img, combined, zmap_image

def process_quick_zmap(ci, sigma, threshold):
    # Blur CI
    blurred_ci = gaussian_filter(ci, sigma=sigma, mode='constant', cval=0)
    # Create z-map
    scaled_image = (blurred_ci - np.mean(blurred_ci)) / np.std(blurred_ci)
    # Apply threshold
    zmap = scaled_image.copy()
    zmap[(zmap > -threshold) & (zmap < threshold)] = np.nan
    return blurred_ci, scaled_image, zmap

# TODO I can generate the zmaps outside of generating the CIs...
# TODO here I need to make sure that I do the stimuli_params[selection] and also pass responses
# TODO to make sure the params are in the correct order
def process_ttest_zmap(params, responses, patches, patch_idx, img_size, ci):
    responses = responses.reshape((responses.shape[0], 1))
    weighted_parameters = params * responses
    n_observations = len(responses)
    noise_images = np.zeros((img_size, img_size, n_observations))
    # For each observation, generate noise image
    for obs in range(n_observations):
        noise_images[:, :, obs] = generate_noise_image(weighted_parameters[obs], patches, patch_idx)
    # Apply t-test along the observation axis directly
    t_stat, p_values = ttest_1samp(noise_images, popmean=0, axis=2)
    # Create Z-map
    zmap = np.sign(ci) * np.abs(norm.ppf(p_values / 2))
    return zmap

#TODO This is probably not needed, since for multiple participants we can do it outside the loop or in a different function
#TODO think of good descriptive names for variables
def generate_ci_noise(stimuli, responses, patches, patchIdx):
    # normalize responses, so we can multiply correctly
    responses = responses.reshape((responses.shape[0],1))
    # below original function
    weighted = stimuli * responses
    if weighted.ndim == 1:
        params = weighted
    else:
        params = weighted.mean(axis=0)
    return generate_noise_image(params, patches, patchIdx)

# TODO mask can be a masked array loaded from file or directly fed into the function
# TODO split load_mask_from_file and add to pipe
def apply_mask(ci, mask):
    # mask is a path to a file
    if isinstance(mask, str):
        mask_matrix = np.array(Image.open(mask).convert('L'))
    elif isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_matrix = mask
    else:
        raise ValueError("The mask argument is neither a string nor a 2D matrix!")

    if mask_matrix.shape != ci.shape:
        raise ValueError(f"Mask is not of the same dimensions as the CI: {ci.shape} vs {mask_matrix.shape}")

    if np.any((mask_matrix != 0) & (mask_matrix != 255)):
        raise ValueError("The mask contains values other than 0 (black) or 255 (white)!")
    
    masked_ci = np.where(mask_matrix == 0, np.nan, ci)
    return masked_ci


def apply_scaling(base, ci, scaling, constant):
    if scaling == 'none':
        scaled = ci
    elif scaling == 'constant':
        scaled = (ci + constant) / (2 * constant)
        if np.any((scaled > 1.0) | (scaled < 0)):
            print("Chosen constant value for constant scaling made noise "
                  "of classification image exceed possible intensity range "
                  "of pixels (<0 or >1). Choose a lower value, or clipping "
                  "will occur.")
    elif scaling == 'matched':
        min_base = np.min(base)
        max_base = np.max(base)
        min_ci = np.min(ci[~np.isnan(ci)])
        max_ci = np.max(ci[~np.isnan(ci)])
        scaled = min_base + ((max_base - min_base) * (ci - min_ci) / (max_ci - min_ci))
    elif scaling == 'independent':
        constant = max(abs(np.nanmin(ci)), abs(np.nanmax(ci)))
        scaled = (ci + constant) / (2 * constant)
    else:
        print(f"Scaling method '{scaling}' not found. Using none.")
        scaled = ci
    return scaled

def combine(scaled, base):
    return (scaled + base) / 2

def save_image(image, path, clip=True, scale=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if clip:
        image = np.clip(image, 0, 1)
    if scale:
        image = image * 255
    Image.fromarray(image.astype(np.uint8)).save(path)
    #image.save(os.path(path))
    #Image.fromarray(image.astype(np.uint8)).save(os.path(path))

def save_to_image(baseimage, combined, targetpath, filename, anti_CI):
    if anti_CI:
        filename = 'python_antici_' + filename
    else:
        filename = 'python_ci_' + filename
    filename += '.png'
    os.makedirs(targetpath, exist_ok=True)
    Image.fromarray((combined * 255).astype(np.uint8)).save(os.path.join(targetpath, filename))



#TODO This is jittable and equivalent with jnp equivalent
#TODO it doesn't play nice with saving images
#@jit
def generate_noise_image(params, patches, patchIdx):
    # we need to convert to int and subtract 1 to make it 0-indexed
    patch_indices = patchIdx.astype(int)
    pd = patches.shape
    patch_params = np.array(params[patch_indices]).reshape(pd)
    reshaped_matrix = (patches * patch_params).reshape((pd[0]*pd[1], pd[2]))
    noise = np.mean(reshaped_matrix, axis=1).reshape(pd[0:2])
    return noise

#TODO better name
def generate_meshgrid(cycles, patch_size):
    x = np.linspace(0, cycles, patch_size)
    y = np.linspace(0, cycles, patch_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def generate_sinusoid(patch_size: int, cycles: float, angle: float, phase: float, contrast: float):
    X, Y = generate_meshgrid(cycles, patch_size)
    angle = math.radians(angle)
    sinepatch = X * math.cos(angle) + Y * math.sin(angle)
    sinusoid = (sinepatch * 2 * math.pi) + phase
    sinusoid = contrast * np.sin(sinusoid)
    return sinusoid

def generate_gabor(patch_size, cycles, angle, phase, sigma, contrast):
    sinusoid = generate_sinusoid(patch_size, cycles, angle, phase, contrast)
    x0 = np.linspace(-0.5, 0.5, patch_size)
    X, Y = np.meshgrid(x0, x0)
    gauss_mask = np.exp(-((X ** 2 + Y ** 2) / (2 * (sigma / patch_size) ** 2)))
    gabor = gauss_mask * sinusoid
    return gabor

@partial(jit, static_argnames=['img_size'])
def generate_scales(img_size:int = 512):
    x, y = jnp.meshgrid(jnp.arange(start=1, stop=img_size+1, step=1), jnp.arange(start=1, stop=img_size+1, step=1))
    patch_size = x / y
    #TODO scales will be integers, so make sure to check that all are integer convertible
    #and then actually convert it to integers.
    patch_size_int = np.round(patch_size).astype(int)
    return patch_size_int

def generate_noise_pattern(img_size=512, nscales=5, noise_type='sinusoid', sigma=25):
    #TODO move those to variables
    # Settings of sinusoids
    orientations = np.array([0, 30, 60, 90, 120, 150])
    phases = np.array([0, np.pi/2])
    scales = 2 ** np.arange(nscales)
    assert scales.dtype == np.int64

    # Size of patches per scale
    patch_sizes = generate_scales(img_size=img_size)

    # Number of patch layers needed
    nr_patches = len(scales) * len(orientations) * len(phases)

    # Allocate memory
    patches = np.zeros((img_size, img_size, nr_patches))
    patch_idx = np.zeros((img_size, img_size, nr_patches))

    co = 0  # patch layer counter
    idx = 0  # contrast index counter

    for scale in scales:
        # iterate over each scale (i.e. 512, 256, 128, 64, 32)
        patch = patch_sizes[scale - 1, img_size - 1]
        for orientation in orientations:
            for phase in phases:
                if noise_type == 'gabor':
                    p = generate_gabor(patch, 1.5, orientation, phase, sigma, 1)
                else:
                    # img_size, cycles, angle, phase, contrast
                    p = generate_sinusoid(patch, 2, orientation, phase, 1)
                # Repeat to fill scale
                patches[:, :, co - 1] = np.tile(p, (scale, scale))
                # Create index matrix
                for col in range(1, scale + 1):
                    for row in range(1, scale + 1):
                        # Insert absolute index for later contrast weighting
                        patch_idx[patch * (row - 1) : patch * row, patch * (col - 1):patch * col, co - 1] = idx
                        # Update contrast counter
                        idx += 1
                # Update layer counter
                co += 1
    return patches, patch_idx, noise_type

def generate_stimuli_params(n_trials, nscales):
    # Compute number of parameters needed
    nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
    # generate stimuli params for the trial
    stimuli_params = np.random.uniform(-1, 1, size=(n_trials, nparams))
    return stimuli_params

def generate_stimuli_noise(n_trials, nscales, img_size, noise_type, sigma):
    stimuli_params = generate_stimuli_params(n_trials, nscales)
    # Preallocate stimulus array
    stimuli = np.zeros((n_trials, img_size, img_size))
    # Generate noise pattern
    patches, patch_idx, noise_type = generate_noise_pattern(img_size=img_size, noise_type=noise_type, nscales=nscales, sigma=sigma)
    for trial in tqdm(range(n_trials)):
        #for base_face in base_faces:
        params = stimuli_params[trial]
        noise_pattern = generate_noise_image(params, patches, patch_idx)
        stimuli[trial,:,:] = noise_pattern
    return stimuli, patches, patch_idx
    
def read_image(filename, grayscale=True, maximize_contrast=True):
    img = Image.open(filename)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    if maximize_contrast:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.asarray(img)

def generate_stimuli_2IFC(img, n_trials=770, img_size=512, stimulus_path='./stimuli', label='rcic', seed=1, maximize_baseimage_contrast=True, noise_type='sinusoid', nscales=5, sigma=25, return_as_dataframe=False, save_as_png=True, save_rdata=True):
    # Initialize
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(stimulus_path, exist_ok=True)
    base_face = img
    stimuli, patches, patch_idx = generate_stimuli_noise(n_trials, nscales, img_size, noise_type, sigma)
    for trial in tqdm(range(stimuli.shape[0])):
        trial_adj = trial+1
        # Scale noise
        stimulus = ((stimuli[trial, :, :] + 0.3) / 0.6)
        # Add base face
        combined = (stimulus + base_face) / 2.0

        filename_ori = f"{label}_aName_{seed:01d}_{trial_adj:05d}_p_ori.png"
        save_image(combined, "./stimuli/"+filename_ori)

        # Compute inverted stimulus
        stimulus = ((-stimuli[trial, :, :] + 0.3) / 0.6)
        # Add base face
        combined = (stimulus + base_face) / 2.0

        filename_inv = f"{label}_aName_{seed:01d}_{trial_adj:05d}_p_inv.png"
        save_image(combined, "./stimuli/"+filename_inv)
    
    #TODO get version from package itself
    # generator_version = '0.4.0'
    if save_rdata:
        timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M")
        filename_rdata = f"{label}_seed{seed}_time_{timestamp}"

        np.savez(os.path.join(stimulus_path, filename_rdata), 
                 patches=patches,
                 patchIdx=patch_idx,
                 noise_type=noise_type)
    
    if return_as_dataframe:
        return stimuli
    
# Usage
n_trials = 5
img_size = 512
nscales = 5
sigma = 25
# nparams = sum(6 * 2 * np.power(2, np.arange(nscales))**2)
# params = np.random.uniform(-1, 1, size=(n_trials, nparams))
noise_type = 'sinusoid'
#base_face_files_dict = {'aName': 'base_face.jpg'}
image = img = read_image(os.getcwd()+"/rcpyci/"+"base_face.jpg", grayscale=(noise_type == 'sinusoid'))
result_python_unconv = generate_stimuli_2IFC(img=image,
                                            n_trials=n_trials, 
                                            img_size=img_size, 
                                            noise_type=noise_type,
                                            nscales=nscales, 
                                            sigma=sigma,
                                            return_as_dataframe=True)

print(result_python_unconv.shape)

stimuli = np.arange(n_trials)
responses = np.ones(shape=(n_trials,))
ci, scaled, base, combined, zmap_image = generate_CI(stimuli=stimuli,
                            responses=responses,
                            img=image, 
                            anti_CI=False,
                            sigma=sigma,
                            zmap=True,
                            zmapmethod='t.test',
                            threshold=1)



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

# Paths to the folders containing PNG files
folder1 = "stimuli_ref"
folder2 = "stimuli"

# Compare images in folders
compare_images_in_folders(folder1, folder2)

print("COMPARING GENERATED CI")
compare_images('/home/hval/rcpyci/cis/python_ci_compared.png','/home/hval/rcpyci/cis/python_ci_.png')

zmap = np.load("/home/hval/rcpyci/zmap/zmap_comp.npy")
if np.array_equal(zmap_image,zmap):
    print('ZMAP t.test are equal also')
else:
    raise KeyError