import os

import numpy as np
from PIL import Image


def autoscale(cis, save_as_pngs=True, targetpath='./cis'):
    # Get range of each ci
    ranges = np.zeros((len(cis), 2))
    for i, ciname in enumerate(cis.keys()):
        ranges[i, :] = [np.min(cis[ciname]['ci']), np.max(cis[ciname]['ci'])]

    # Determine the lowest possible scaling factor constant
    if abs(np.min(ranges[:, 0])) > np.max(ranges[:, 1]):
        constant = abs(np.min(ranges[:, 0]))
    else:
        constant = np.max(ranges[:, 1])

    print(f'Using scaling factor constant: {constant}')

    # Scale all noise patterns
    for ciname, data in cis.items():
        cis[ciname]['scaled'] = (data['ci'] + constant) / (2 * constant)

        # Combine and save to PNG if necessary
        if save_as_pngs:
            ci = (cis[ciname]['scaled'] + data['base']) / 2
            os.makedirs(targetpath, exist_ok=True)

            img = Image.fromarray((ci * 255).astype('uint8'))
            img.save(os.path.join(targetpath, f'{ciname}_autoscaled.png'))

    return cis

# Example usage
# cis = {'ci1': {'ci': ci1, 'base': base1}, 'ci2': {'ci': ci2, 'base': base2}}
# autoscale_result = autoscale(cis, save_as_pngs=True, targetpath='./cis')
