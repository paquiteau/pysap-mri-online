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
2. Ensure to have OpenCL/ Cuda installed and setup on your machine

``` sh
sudo apt install nfft # required for pynfft
pip install -r requirements.txt 
pip install --upgrade git+https://github.com/CEA-COSMIC/pysap.git
pip install --upgrade git+https://github.com/paquiteau/pysap-mri.git
```

# Known Problems:
Gradient operation with GPU cause segfault with PyNUFFT. This is likely due to miss-feeding an array.


