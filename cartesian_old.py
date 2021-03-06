""" Based on "online_gpu" branch
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.

DO NOT RUN, present in repo only for ease of access, adapted version is online_reconstruction.py

"""

# Package import
import pysap
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri_online.gradient import Grad2D_pMRI
from pysap.plugins.mri.reconstruct.fourier import FFT2
from pysap.plugins.mri.parallel_mri_online.linear import Pywavelet2
from pysap.plugins.mri.parallel_mri.proximity import Threshold
from pysap.plugins.mri.parallel_mri_online.utils import compute_ssim

# Third party import
import os
import time
import numpy as np
import scipy.fftpack as pfft
import matplotlib.pyplot as plt
from twixreader import Twix
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing

# Loading input data
folder = '/neurospin/optimed/LoubnaElGueddari/Data/online_2D/single_channel_cartesian/2017-04-04_sparkling2D_paper/'
data_path = folder + 'meas_MID276_CSGRE_ref_OS1_FID10929.dat'
twix = Twix(data_path)
kspace_ref = np.squeeze(twix[0]['ima'].raw()).reshape((32, 512, 512))
NEX = 20
kspace_ref = np.mean(kspace_ref[:NEX, :, :], axis=0)
I = pfft.ifftshift(pfft.ifft2(pfft.fftshift(kspace_ref))).astype("complex128")
idx_seq = np.load(folder + 'cartesian_idx_sequence_1.npy')
mask = np.load(folder + "cartesian_line_UF_2.npy")
im_mask = np.zeros(I.shape)
im_mask[np.abs(I) > 3e-08] = 1


def getLargestCC(segmentation):
    labels = label(segmentation, background=0)
    bincounts = np.bincount(labels.flat)
    idx = np.argsort(bincounts)[::-1]
    largestCC = np.copy(labels == idx[1])
    return largestCC


im_mask = binary_closing(getLargestCC(im_mask), np.ones((5, 5)))
b_size = 88
t_it = 0.100
TR = 0.550
batches = np.arange(b_size, 176 + b_size, b_size)  # [176]
if batches.shape[0] > 1:
    nb_it_batches = [np.floor(b_size * 1.0 * TR / t_it).astype('int') for _ in
                     range(batches.shape[0] - 1)]
    nb_it_batches.append(200)
else:
    nb_it_batches = [200]

directory = folder + 'online_reconstrcution_batch_size_' + str(b_size)

dir_exist = False
try:
    os.stat(directory)
    dir_exist = True
except:
    dir_exist = False
    os.mkdir(directory)
    os.mkdir(directory + '/masks_batches/')
    os.mkdir(directory + '/images_batches/')
    os.mkdir(directory + '/times_batches/')
if dir_exist:
    try:
        os.stat(directory + '/masks_batches/')
    except:
        os.mkdir(directory + '/masks_batches/')
    try:
        os.stat(directory + '/images_batches/')
    except:
        os.mkdir(directory + '/images_batches/')
    try:
        os.stat(directory + '/times_batches/')
    except:
        os.mkdir(directory + '/times_batches/')
#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.


# Generate the subsampled kspace
kspace = mask * pfft.ifftshift(pfft.fft2(pfft.fftshift(I)))

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask)

linear_op = Pywavelet2("sym8", nb_scale=4, multichannel=True)
coeffs = linear_op.op(np.expand_dims(I, axis=0))
mu = 1e-8 / 176
prox_dual_op = Threshold(weights=mu)

fourier_op = FFT2(kspace_loc, shape=I.shape)
gradient_op = Grad2D_pMRI(data=np.expand_dims(np.zeros_like(kspace), axis=0),
                          fourier_op=fourier_op,
                          gradient_spec_rad=1.1)
gradient_op.fourier_op._mask = np.zeros_like(mask)
# Start the POGM reconstruction
# max_iter = 200

from modopt.opt.linear import Identity
from modopt.opt.algorithms import Condat

prox_op = Identity()

# Define the optimizer
sigma = 0.5
eps = 5e-8
norm = 1.0  # linear_op.l2norm(np.expand_dims(I, axis=0).shape)
tau = 1.0 / (gradient_op.spec_rad / 2 + sigma * norm ** 2 + eps)
opt = Condat(
    x=np.zeros((1, *fourier_op.shape), dtype="complex128"),
    y=np.zeros_like(coeffs),
    grad=gradient_op,
    prox=prox_op,
    prox_dual=prox_dual_op,
    linear=linear_op,
    cost=None,
    rho=1.0,
    sigma=sigma,
    tau=tau,
    rho_update=None,
    sigma_update=None,
    tau_update=None,
    auto_iterate=False)
final_ssim = []
time_batches = []
time_batches.append(batches[0] * TR)
final_ssim.append(0)
final_ssim = np.asarray(final_ssim)
final_cost = []
final_cost.append(np.sum(np.abs(opt._grad.op(np.zeros_like(opt._x_old)) -
                                kspace_ref) ** 2) +
                  opt._prox_dual.weights * np.sum(
    np.abs(opt._linear.op(np.zeros_like(opt._x_old)))))
final_cost = np.asarray(final_cost)
time_batches = np.asarray(time_batches)
fig, ax = plt.subplots()
for idx, batch in enumerate(batches):
    obs_data = np.copy(opt._grad.obs_data)
    obs_data[:, :, idx_seq[:batch]] = kspace_ref[:, idx_seq[:batch]]
    opt._grad = Grad2D_pMRI(data=obs_data,
                            fourier_op=fourier_op,
                            gradient_spec_rad=1.1)
    opt._grad.fourier_op._mask[:, idx_seq[:batch]] = np.ones((I.shape[0], batch))
    cost_func_batch = []
    ssim_batches = []
    print(batch, opt._prox_dual.weights)
    opt._prox_dual.weights = mu * batch
    time_it = []
    # cost_func_batch.append()
    for _ in range(nb_it_batches[idx]):
        start = time.time()
        opt._update()
        stop = time.time()
        time_it.append(stop - start)
        cost_func_batch.append(np.sum(np.abs(opt._grad.op(opt._x_new) -
                                             kspace_ref) ** 2) +
                               opt._prox_dual.weights * np.sum(np.abs(opt._y_new)))
        ssim_batches.append(compute_ssim(np.fft.fftshift(np.squeeze(opt._x_new)), I))
    np.save(directory + '/times_batches/time_batch_nb_' + str(idx), time_it)
    #  Saving batch mask
    ax.imshow(opt._grad.fourier_op._mask, cmap='gray');
    ax.axis('off')
    imsave_path = directory + '/masks_batches/mask_batch_{0:03d}.png'.format(batch)
    plt.imsave(fname=imsave_path, arr=opt._grad.fourier_op._mask)
    #  Saving images
    np.save(directory + '/images_batches/image_batch_nb_{0:03d}'.format(idx), opt._x_new)
    ax.imshow(np.fft.fftshift(np.squeeze(np.abs(opt._x_new))), cmap='gray');
    ax.axis('off')
    imsave_path = directory + '/images_batches/I_batch_{0:03d}.png'.format(batch)
    plt.imsave(fname=imsave_path, arr=np.fft.fftshift(np.squeeze(np.abs(opt._x_new))), cmap='gray')
    # Updating cost function
    time_batches = np.concatenate((time_batches, np.asarray(time_it)))
    final_cost = np.concatenate((final_cost, np.asarray(cost_func_batch)))
    final_ssim = np.concatenate((final_ssim, np.asarray(ssim_batches)))

np.save(directory + '/resume_time.npy', np.asarray(time_batches))
np.save(directory + '/resume_cost.npy', final_cost)
np.save(directory + '/resume_ssim.npy', final_ssim)
plt.figure()
plt.plot(final_cost)
plt.figure()
plt.plot(final_ssim)
plt.figure()
plt.imshow(np.fft.fftshift(np.squeeze(np.abs(opt._x_old))))
plt.show()
