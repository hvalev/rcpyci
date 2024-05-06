# Reference implementation of the R rcicr package in python
TBD

### Mapping of R functions into python reference implementation
| Original R Script                  | Converted Python Script                |
|------------------------------------|----------------------------------------|
| autoscale.R                        | autoscale.py                           |
| batchGenerateCI.R                  | batchGenerateCI.py                     |
| batchGenerateCI2IFC.R              | batchGenerateCI2IFC.py                 |
| computeCumulativeCICorrelation.R   | computeCumulativeCICorrelation.py      |
| computeInfoVal2IFC.R               | computeInfoVal2IFC.py                  |
| deg2rad.R                          | deg2rad.py                             |
| generateCI.R                       | generateCI.py                          |
| generateCI2IFC.R                   | generateCI2IFC.py                      |
| generateCINoise.R                  | generateCINoise.py                     |
| generateGabor.R                    | generate_noise.py                      |
| generateNoiseImage.R               | generateNoiseImage.py                  |
| generateNoisePattern.R             | generate_noise.py                      |
| generateReferenceDistribution.R    | generateReferenceDistribution2IFC.py   |
| generateSinusoid.R                 | generate_noise.py                      |
| generateStimuli2IFC.R              | generateStimuli2IFC.py                 |
| plotZmap.R                         |                                        |
| simulateNoiseIntensities.R         | simulateNoiseIntensities.py            |
| zzz.R                              |                                        | 

### Requirements
In order to run the tests of the reference implementation against the original R implementation, the package rpy2 is needed. TBD complete this information