#!/bin/bash
##export LD_LIBRARY_PATH=/usr/local/lib/R/modules:/usr/local/lib/R/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/R/lib:/usr/local/lib/R/modules
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/R/lib
export R_HOME=/usr/local/lib/R
# needed for compiling devtools. Mostly included here for redundancy
apt update && apt-get install nano libgdal-dev python3.11-venv -y
# for rpy2
apt-get install python3-dev -y
# compiling rcicr. Mostly included for redundancy
R -e "devtools::install('/tests/rcicr', repos='http://cran.rstudio.com/')"