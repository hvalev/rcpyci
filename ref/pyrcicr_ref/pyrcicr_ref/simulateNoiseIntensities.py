
import matplotlib.pyplot as plt
import numpy as np
from generate_noise import generate_noise_pattern
from generateNoiseImage import generate_noise_image
from tqdm import tqdm


def simulate_noise_intensities(nrep=1000, img_size=512):
    results = np.zeros((nrep, 2))
    s = generate_noise_pattern(img_size)

    for i in tqdm(range(nrep), ncols=100, desc="Simulating Noise Intensities"):
        params = (np.random.rand(4096) * 2) - 1

        noise = generate_noise_image(params, s)
        results[i, :] = [np.min(noise), np.max(noise)]

    plt.boxplot(results)
    plt.show()

    return results

# Example usage
results = simulate_noise_intensities(nrep=1000, img_size=512)