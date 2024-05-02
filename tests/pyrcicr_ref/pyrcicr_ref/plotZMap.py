import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plotZmap(zmap, bgimage='', sigma=0, threshold=3, mask=None, decoration=True, targetpath='zmaps', filename='zmap', size=512, **kwargs):
    # Create target directory
    os.makedirs(targetpath, exist_ok=True)

    # Apply threshold
    zmap[np.abs(zmap) < threshold] = np.nan

    # Plot with decoration
    if decoration:
        fig, ax = plt.subplots()
        ax.set_title(f'Z-map of {filename}')
        ax.set_xlabel(f'sigma = {sigma}, threshold = {threshold}')
        cmap = plt.get_cmap("viridis")
        cmap.set_bad(color='white')
        ax.imshow(zmap, cmap=cmap, extent=[0, 1, 0, 1])

        # Add bgimage if specified
        if bgimage:
            bgimg = Image.open(bgimage)
            ax.imshow(bgimg, extent=[0, 1, 0, 1])

        if mask is not None:
            ax.imshow(mask, cmap='gray', alpha=0.5, extent=[0, 1, 0, 1])

        # Save the plot as a PNG image
        plt.savefig(os.path.join(targetpath, f'{filename}.png'), dpi=size/4, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    # Without decoration
    else:
        fig, ax = plt.subplots()
        ax.axis('off')

        if bgimage:
            bgimg = Image.open(bgimage)
            ax.imshow(bgimg)

        ax.imshow(zmap, cmap='viridis', extent=[0, 1, 0, 1], alpha=0.7)

        if mask is not None:
            ax.imshow(mask, cmap='gray', alpha=0.5, extent=[0, 1, 0, 1])

        plt.savefig(os.path.join(targetpath, f'{filename}.png'), dpi=size/4, bbox_inches='tight', pad_inches=0.02)
        plt.close()

# Example usage:
# plotZmap(zmap, bgimage='base_image.png', sigma=0.5, threshold=2, mask=None, decoration=True, targetpath='zmaps', filename='zmap', size=512)
