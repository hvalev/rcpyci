import math
import numpy as np

# def generate_sinusoid(size, frequency, orientation, phase, contrast):
#     x = np.linspace(0, size - 1, size)
#     y = np.linspace(0, size - 1, size)
#     [X, Y] = np.meshgrid(x, y)
    
#     angle = orientation * math.pi / 180.0
#     xt = X * math.cos(angle) + Y * math.sin(angle)
#     yt = -X * math.sin(angle) + Y * math.cos(angle)
    
#     return contrast * np.sin(2 * math.pi * frequency * xt + phase)

def generate_sinusoid(img_size: int, cycles: int, angle: int, phase: int, contrast: float):
    angle = math.radians(angle)
    x = np.linspace(0, cycles, img_size)
    y = np.linspace(0, cycles, img_size)
    X, Y = np.meshgrid(x, y)

    sinepatch = X * math.cos(angle) + Y * math.sin(angle)
    sinusoid = (sinepatch * 2 * math.pi) + phase
    sinusoid = contrast * np.sin(sinusoid)
    
    return sinusoid

# def generate_gabor(size, orientation, phase, sigma, frequency, contrast):
#     x = np.linspace(0, size - 1, size)
#     y = np.linspace(0, size - 1, size)
#     [X, Y] = np.meshgrid(x, y)
    
#     angle = orientation * math.pi / 180.0
#     xt = X * math.cos(angle) + Y * math.sin(angle)
#     yt = -X * math.sin(angle) + Y * math.cos(angle)
    
#     spatial_frequency = 1.0 / frequency
#     wave = np.exp(-0.5 * (xt ** 2 + yt ** 2) / sigma ** 2) * np.cos(2 * math.pi * spatial_frequency * xt + phase)
    
#     return contrast * wave

def generate_gabor(img_size, cycles, angle, phase, sigma, contrast):
    sinusoid = generate_sinusoid(img_size, cycles, angle, phase, contrast)
    x0 = np.linspace(-0.5, 0.5, img_size)
    X, Y = np.meshgrid(x0, x0)

    gauss_mask = np.exp(-((X ** 2 + Y ** 2) / (2 * (sigma / img_size) ** 2)))
    gabor = gauss_mask * sinusoid

    return gabor

def generate_scales(img_size=512, nscales=5):
    scales = 2 ** np.arange(nscales)
    x, y = np.meshgrid(np.arange(1, img_size+1), np.arange(1, img_size+1))
    patch_size = x / y
    #TODO scales will be integers, so make sure to check that all are integer convertible
    #and then actually convert it to integers.
    return patch_size

def generate_noise_pattern(img_size=512, nscales=5, noise_type='sinusoid', sigma=25, pre_0_3_0=False):
    # Settings of sinusoids
    orientations = np.array([0, 30, 60, 90, 120, 150])
    phases = np.array([0, np.pi/2])
    scales = 2 ** np.arange(nscales)

    # Size of patches per scale
    #patch_size = np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1), np.arange(1, len(scales) + 1))[0] / np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1), np.arange(1, len(scales) + 1))[1]
    patch_size = generate_scales(img_size=img_size)
    
    # Number of patch layers needed
    nr_patches = len(scales) * len(orientations) * len(phases)

    # Preallocate memory
    patches = np.zeros((img_size, img_size, nr_patches))
    patch_idx = np.zeros((img_size, img_size, nr_patches))

    # Counters
    if pre_0_3_0:
        co = 0  # patch layer counter
        idx = 0  # contrast index counter
    else:
        co = 1  # patch layer counter
        idx = 1  # contrast index counter

    for scale in scales:
        for orientation in orientations:
            for phase in phases:
                # Generate single patch
                #size = patch_size[int(scale) - 1, :, :]
                size = patch_size[int(scale) - 1, img_size - 1]
                if noise_type == 'gabor':
                    p = generate_gabor(int(size), 1.5, int(orientation), phase, sigma, 1)
                else:
                    # img_size, cycles, angle, phase, contrast
                    p = generate_sinusoid(int(size), int(2), int(orientation), phase, 1)

                # Repeat to fill scale
                patches[:, :, co - 1] = np.tile(p, (scale, scale))

                # Create index matrix
                for col in range(1, scale + 1):
                    for row in range(1, scale + 1):
                        # Insert absolute index for later contrast weighting
                        patch_idx[int(size * (row - 1)):int(size * row), int(size * (col - 1)):int(size * col), co - 1] = idx

                        # Update contrast counter
                        idx += 1

                # Update layer counter
                co += 1

    return {'patches': patches, 'patchIdx': patch_idx, 'noise_type': noise_type, 'generator_version': '0.3.0'}

