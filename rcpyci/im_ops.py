### image operators


### pipelines
import jax.numpy as jnp
from jax import jit
from jax.lax import conv

# def grayscale(rgb_image):
#     # Convert RGB image to grayscale using luminance method
#     return jnp.dot(rgb_image, jnp.array([0.2989, 0.5870, 0.1140]))

# def resize(image, new_size):
#     # Simple resize operation using nearest neighbor interpolation
#     return jnp.array(Image.resize(image, new_size))

# def gaussian_blur(image, sigma=1):
#     # Apply 2D Gaussian blur using convolution
#     kernel_size = int(2 * round(2 * sigma) + 1)
#     kernel = jnp.exp(-jnp.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)**2 / (2. * sigma**2))
#     kernel = kernel / jnp.sum(kernel)
#     return conv(image, kernel[jnp.newaxis, :], (1,))  # Applying convolution along the single channel





# Define a pipeline function that chains multiple image operations
def image_pipeline(image):
    grayscale_image = grayscale(image)
    resized_image = resize(grayscale_image, (100, 100))
    blurred_image = gaussian_blur(resized_image, sigma=1)
    return blurred_image

# JIT compile the pipeline function for performance
jit_image_pipeline = jit(image_pipeline)

# Now you can use the compiled function on your image data
processed_image = jit_image_pipeline(input_image)