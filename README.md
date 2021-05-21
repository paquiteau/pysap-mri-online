[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/paquiteau/projet-atsi/master)

# Online MRI Reconstruction

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

# TODO: 
## Parameters and Hyper-parameters
### Inputs parameters
 - Mono- vs Multi-coil MRI
 - Size of the reconstruction
 - Size of the kspace acquision
 - Acquisition strategy of the kspace 

### Reconstruction Operator
 - Fourier Operator: FFT2D, FFT1.5D, NUFFT
 - Regularisation Domain: Wavelet(type, nb_scale)... 
 - Regularisation function: 
   - Norm l_1 l_2 l_{1,2}, OWL
   - Order of the dimension (per coil, per bands, all coef...)
 - Gradient formulation: Analysis vs Synthesis
### Reconstruction Algorithm
#### Forward-Backward
-> Regularisation at which step
- POGM 
- FISTA
- Vanilla Gradient Descent
- Custom Gradient Descent Strategies

#### Primal-Dual 
-> regularisation mandatory at each step (use of the dual variable)

- Condat Vu 