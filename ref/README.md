# Equivalence testing between python and R's implementation of rcicr
This page contains details on the implementation of `pyrcicr_ref` which allows for equivalence testing between the python and the original R implementation of [rcicr](https://github.com/rdotsch/rcicr/).

## Begin
In order to functionally test whether two implementations are same or near-similar, they need to be evaluated piece by piece or in this context, function by function. Luckily, R is, at its heart, a language based in functional programming, very similar to python which makes this task easier.

## Environment
In order to achieve reproducible results an environment is needed where we can run both python and R code and we can also pass around objects between R and python. The `rpy2` package allows to convert python and numpy objects into R's equivalent representations and vice-versa and is central to this effort. The rest is a complete R environment and some staple python packages such as numpy, scipy and the likes. This environment can be build by building a docker image `Dockerfile.test` from this folder. Alternatively, a pre-build image is published @ [docker hub](https://hub.docker.com/repository/docker/hvalev/rcpyci/general).

## Modifications to rcicr
Some slight modifications have been made to the original [rcicr] package. The package (along with those modifications) has been included in this repository for reproducibility. The changes have been documented with inline comments in the codebase. Those include mostly changes which allow us to retrieve data from a function or allow us to insert some partial data in a function in lieu of the function computing it herself. This allows to substitute data otherwise generated using seeded random values with a fixed equivalent to ultimately compare function outputs for equivalence.

## Running the test suite
To run the test suite a single script simply run `./run_tests.sh` from within the `./ref/` folder. This will 1) build the environment required to execute R and python code 2) install the modified rcicr library and necessary python packages and 3) run all the tests and report the results to the terminal. Building the image may take between 10 to 20 minutes, depending on your hardware and internet speed. Running the tests could take up to 2 minutes.



## Further reading
R uses a column major ordering of its arrays (fortran style), while python uses a row major ordering of its arrays (c style). This means that sometimes arrays need to be reshaped before comparing them. Luckily, in python this is a very simple operation as follows:
`result_p = result_python.reshape(result_r.shape, order='F')`
You can find a more comprehensive example in the test suite for the reference implementation.