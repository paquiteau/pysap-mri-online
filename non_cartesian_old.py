""" Based on "online_gpu" branch

DO NOT RUN, present in repo only for ease of access, adapted version is online_reconstruction.py

"""

# Package import
import pysap
from pysap.plugins.mri.parallel_mri_online.gradient import Grad2D_pMRI
from pysap.plugins.mri.reconstruct.fourier import NFFT2
from pysap.plugins.mri.reconstruct_3D.fourier import NUFFT
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.parallel_mri_online.proximity import OWL, NuclearNorm
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
folder = '/neurospin/optimed/LoubnaElGueddari/Data/online_2D/multi_channel_non_cartesian/2017-09-18_wilson_complete/'
data_path = folder + 'meas_MID111_CSGRE_ref_N512_FID24209.npy'
folder += 'sparkling_nc34x3072/'
kspace_ref = np.load(data_path)
Il = pfft.ifftshift(pfft.ifft2(pfft.fftshift(kspace_ref), axes=[1, 2])).astype("complex128")
print(Il.shape)
I = np.sqrt(np.sum(np.abs(Il) ** 2, axis=0))
samples = np.load(folder + "GA_GradientFile_Samples_SPARKLING_N512_R3_nc34x3073_Dt10us_densitytrick_BB_upsampled_Smax302.npy")
samples = np.reshape(samples, (34 * 3072, 2))
im_mask = np.zeros(I.shape)
im_mask[np.abs(I) > 3e-08] = 1
def getLargestCC(segmentation):
    labels = label(segmentation, background=0)
    bincounts = np.bincount(labels.flat)
    idx = np.argsort(bincounts)[::-1]
    largestCC = np.copy(labels == idx[1])
    return largestCC
im_mask = binary_closing(getLargestCC(im_mask), np.ones((5,5)))
b_size = 34
t_it = 4
TR = 0.550
batches = np.arange(b_size, 34 +1, b_size)#[34]
print(batches)
if batches.shape[0] > 1:
    nb_it_batches = [np.floor(b_size * 1.0 * TR / t_it).astype('int') for _ in
                     range(batches.shape[0] - 1)]
    nb_it_batches.append(200)
else:
    nb_it_batches = [200]

print(nb_it_batches)

directory = folder + 'online_reconstrcution_batch_size_'+ str(b_size)

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

# Generate the subsampled kspace
kspace = np.load(folder + 'GA_meas_MID123_CSGRE_N512_sparkling_nc34_OS2_BR1536_FID24221.npy')

from modopt.opt.linear import Identity
from modopt.opt.algorithms import Condat
prox_op = Identity()
# Get the locations of the kspace samples
linear_op = pyWavelet3("sym8", nb_scale=4, multichannel=True)
coeffs = linear_op.op(Il)
alpha = 1e-05 / 34
beta = 1e-12 / 34
prox_dual_op = OWL(alpha=alpha,
                   beta=beta,
                   bands_shape=linear_op.coeffs_shape,
                   mode='band_based',
                   n_channel=32,
                   num_cores=1)

fourier_op = NUFFT(samples=samples, shape=I.shape, platform='gpu',
                    Kd=(1024, 1024), Jd=(5, 5),
                    n_coils=32, verbosity=0,
                    kspace_mask=np.ones((32, 34*3072)))

gradient_op = Grad2D_pMRI(data=np.zeros_like(kspace.reshape((32, 34 * 3072))),
                          fourier_op=fourier_op)

# Start the POGM reconstruction
# max_iter = 200


# Define the optimizer
sigma = 0.5
eps = 5e-8
norm = 1.0
tau = 1.0 / (gradient_op.spec_rad/2 + sigma * norm**2 + eps)
opt = Condat(
    x=np.zeros((32, *fourier_op.shape), dtype="complex128"),
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
# Setting tome vect
time_batches = np.asarray([TR*b_size])

# Setting SSIM vect
final_ssim = np.asarray([0])

# Setting cost vect
final_cost = np.asarray([np.sum(np.abs(kspace.flatten())**2)])

for idx, batch in enumerate(batches):
    obs_data = np.copy(opt._grad.obs_data)
    obs_data[:, :batch *3072] = np.reshape(kspace[:, :batch, :], (32, batch * 3072))
    opt._grad = Grad2D_pMRI(data=obs_data,
                            fourier_op=fourier_op,
                            gradient_spec_rad=opt._grad.spec_rad)
    opt._grad.fourier_op.kspace_mask[...] = 0
    opt._grad.fourier_op.kspace_mask[:, :batch *3072] = 1
    cost_func_batch = []
    ssim_batches = []
    opt._prox_dual = OWL(alpha=alpha * batch,
                       beta=beta * batch,
                       bands_shape=linear_op.coeffs_shape,
                       mode='band_based',
                       n_channel=32,
                       num_cores=-1)
    print(opt._prox_dual.weights[0])
    time_it = []
    for _ in range(nb_it_batches[idx]):
        print(_)
        start = time.time()
        opt._update()
        stop = time.time()
        time_it.append(stop - start)
        cost_func_batch.append(np.sum(np.abs(opt._grad.op(opt._x_new).flatten() -
                                             kspace.flatten())**2) +
                                opt._prox_dual.cost(opt._linear.op(opt._x_new)))
        ssim_batches.append(compute_ssim(np.sqrt(np.sum(np.abs(opt._x_new)**2,
                                                        axis=0)), I, im_mask))
    time_batches = np.concatenate((time_batches, np.asarray(time_it)))
    np.save(directory + '/times_batches/time_batch_nb_{0:02d}.npy'.format(idx), time_it)

    np.save(directory + '/images_batches/image_batch_nb_{0:02d}.npy'.format(idx), opt._x_new)
    imsave_path = directory + '/images_batches/I_batch_{0:02d}.png'.format(idx)
    plt.imsave(fname=imsave_path,
               arr=np.sqrt(np.sum(np.abs(opt._x_new)**2, axis=0)), cmap='gray')

    # Updating cost function
    final_cost = np.concatenate((final_cost, np.asarray(cost_func_batch)))
    final_ssim = np.concatenate((final_ssim, np.asarray(ssim_batches)))

np.save(directory + '/resume_time.npy', time_batches)
np.save(directory + '/resume_cost.npy', final_cost)
np.save(directory + '/resume_ssim.npy', final_ssim)
plt.figure()
plt.plot(final_cost)
plt.figure()
plt.plot(final_ssim)
plt.figure()
plt.imshow(np.sqrt(np.sum(np.abs(opt._x_new)**2, axis=0)))
plt.show()
