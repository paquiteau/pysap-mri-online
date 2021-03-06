[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/paquiteau/projet-atsi/master)

# Online Reconstruction of 2D MRI

ATSI Master Project 

## Installation 
Dependencies:
 - PySAP
 - PySAP-mri
 - ModOpt
 - And their dependencies.


1. Create a virtual environment (Python 3.6+)
2. Install dependencies:

``` sh
pip install --upgrade modopt
pip install --upgrade git+https://github.com/CEA-COSMIC/pysap.git
pip install --upgrade pysap-mri
```


# Current problems:

PyNUFFT API has changed, and is not supported correctly, to work nicely GPUNUFFT is required, however this requires a CUDA compatible GPU.

