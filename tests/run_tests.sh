#!/bin/bash
# This script compares the original R implementation to the reference python implementation by
# 1) building an environment which allows both the original R and reference python functions to be called,
# 2) their results converted to numpy arrays and compared to one another, and
# 3) running the tests and quantifying the deviations between both implementations
# It is a stepping stone ensuring that the reference implementation is 'close enough'
# to the original R implementation so that it can be used to create a more efficient 
# codebase based on jax allowing cpu/gpu computational backends and leveraging parallelisation
docker build -f Dockerfile.test . --tag test_image
docker run -w /tests test_image /bin/bash -c "export R_HOME=/usr/local/lib/R && export LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib/R/modules && /pyrcicr/bin/python3 test_generate_functions.py && /pyrcicr/bin/python3 test_ci_functions.py"