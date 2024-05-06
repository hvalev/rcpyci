#!/bin/bash
# create the python environment for running the tests
python3 -m venv pyrcicr
source pyrcicr/bin/activate
# install requirements
python3 -m pip install -r /tests/pyrcicr_ref/requirements_rpy2.txt
# install the reference implementation package
python3 -m pip install /tests/pyrcicr_ref/