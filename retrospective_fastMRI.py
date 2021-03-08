"""
Script to carry out Retrospective reconstruction given a set of parameters
on FastMRI data
Author: Chaithya G R
"""
# Package import
from mri.operators import FFT, WaveletN, OWL
from mri.reconstructors import CalibrationlessReconstructor
from mri.scripts.gridsearch import launch_grid

# Third party import
from modopt.opt.proximity import GroupLASSO
from modopt.math.metrics import ssim, psnr, nrmse
from fastmri.data import transforms, subsample, SliceDataset

import operator
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.abspath('./'))
print(sys.path)
import utils
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_loc",
    help="Fast MRI Dataset Location",
    default='/neurospin/optimed/Chaithya/Raw_Data/FastMRI/brain_val'
)
parser.add_argument(
    "--i",
    help="Number of data points",
    type=int,
    default=5,
)
args = parser.parse_args()

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08],
    accelerations=[4]
)
mask_loc = np.load('/neurospin/optimed/Chaithya/Results/Calibrationless/cartesian_right_mask.npy')

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    masked_kspace, mask = transforms.apply_mask(transforms.to_tensor(kspace), mask_func)
    masked_kspace = transforms.tensor_to_complex_np(masked_kspace)
    return masked_kspace, np.squeeze(mask.numpy()), target, filename, slice_num


def recon_data(images, ref_image, filename, slice_num, reg):
    def _metric(test, ref, type):
        def _cropND(img, bounding):
            start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
            end = tuple(map(operator.add, start, bounding))
            slices = tuple(map(slice, start, end))
            return img[slices]
        if type == 'ssim':
            return ssim(_cropND(np.linalg.norm(test, axis=0), ref.shape), ref)
        elif type == 'psnr':
            return psnr(_cropND(np.linalg.norm(test, axis=0), ref.shape), ref)

    metrics = {
        'ssim': {
            'metric': _metric,
            'mapping': {'x_new': 'test', 'y_new': None},
            'cst_kwargs': {'ref': ref_image, 'type': 'ssim'},
            'early_stopping': True,
        },
        'psnr': {
            'metric': _metric,
            'mapping': {'x_new': 'test', 'y_new': None},
            'cst_kwargs': {'ref': ref_image, 'type': 'psnr'},
            'early_stopping': True,
        },
    }
    mask = np.zeros(images.shape[1:])
    mask[:, mask_loc] = 1
    fourier = FFT(
        mask=mask,
        shape=images.shape[1:],
        n_coils=images.shape[0],
    )
    kspace_mask = fourier.op(images)
    linear_params = {
        'init_class': WaveletN,
        'kwargs':
            {
                'wavelet_name': 'sym8',
                'nb_scale': 4,
                'n_coils': kspace_mask.shape[0],
            }
    }
    if reg == 'owl':
        linear_op_temp = WaveletN(**linear_params['kwargs'])
        coeff = linear_op_temp.op(np.zeros_like(kspace_mask))
        regularizer_params = {
            'init_class': OWL,
            'kwargs':
                {
                    'mode': ['all', 'band_based', 'scale_based'],
                    'alpha': np.logspace(-8, -6, 5),
                    'beta': np.logspace(-15, -12, 5),
                    'n_coils': [kspace_mask.shape[0]],
                    'bands_shape': [linear_op_temp.coeffs_shape],
                    'n_jobs': 32,
                }
        }
    else:
        regularizer_params = {
            'init_class': GroupLASSO,
            'kwargs':
                {
                    'weights': np.logspace(-8, -6, 5),
                }
        }
    optimizer_params = {
        'kwargs':
            {
                'optimization_alg': 'condatvu',
                'num_iterations': 200,
                'metrics': metrics,
                'metric_call_period': 50,
            }
    }
    raw_results, test_cases, key_names, best_idx = launch_grid(
        kspace_data=kspace_mask,
        fourier_op=fourier,
        linear_params=linear_params,
        regularizer_params=regularizer_params,
        optimizer_params=optimizer_params,
        reconstructor_class=CalibrationlessReconstructor,
        reconstructor_kwargs={
            'gradient_formulation': 'analysis'
        },
        compare_metric_details={'metric': 'ssim'},
        n_jobs=-1,
        verbose=15,
    )
    test_cases = [t[:-6] for t in test_cases]
    np.save('/neurospin/optimed/Chaithya/Results/Calibrationless/Cartesian/Same_Subsample/' + reg + '/' + filename[:-3] + '_' + str(slice_num) + '.npy',
            (raw_results, test_cases, key_names, best_idx))

data = np.load('/neurospin/optimed/Chaithya/Results/Calibrationless/all_data.npy', allow_pickle=True)[args.i]

recon_data(*data, 'owl')
recon_data(*data, 'gl')