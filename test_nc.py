#%% md

# Non Cartesian online reconstruction

#%%

import os
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy as sp
from mri.operators import FFT,WaveletN, OWL, NonCartesianFFT
from modopt.opt.linear import Identity
from modopt.opt.proximity import GroupLASSO, IdentityProx
from online.operators.proximity import LASSO
from online.generators import Column2DKspaceGenerator,  DataOnlyKspaceGenerator, KspaceGenerator, PartialColumn2DKspaceGenerator
from online.reconstructors import OnlineReconstructor
from online.operators.fourier import ColumnFFT
from project_utils import implot, load_data, create_cartesian_metrics
from online.metrics import ssos, psnr_ssos,ssim_ssos,mse_ssos

plt.rcParams['axes.formatter.useoffset'] = False

plt.style.use({'figure.facecolor':'white'})

#%%

def plot_metric(results, name, *args, log=False, ax=None,**kwargs):
    if ax == None:
        ax = plt.gca()
    if log:
        ax.semilogy(results['metrics'][name]['index'], results['metrics'][name]['values'],*args,**kwargs,label=name)
    else:
        ax.plot(results['metrics'][name]['index'], results['metrics'][name]['values'],*args, **kwargs,label=name)


#%%

DATA_DIR = "data/"
N_JOBS = -1
results = dict()


full_k, real_img, mask_loc, final_mask = load_data(DATA_DIR, 2, monocoil=False)
final_k = np.squeeze(full_k * final_mask[np.newaxis])
square_mask= np.zeros(real_img.shape)
real_img_size = real_img.shape
img_size = [min(real_img.shape)]*2
square_mask[real_img_size[0] // 2 - img_size[0] // 2:real_img_size[0] // 2 + img_size[0] // 2,
            real_img_size[1] // 2 - img_size[1] // 2:real_img_size[1] // 2 + img_size[1] // 2] = 1
trajectories = sp.io.loadmat('data/NCTrajectories.mat')

#%% md

# Offline reconstruction

#%%
print(trajectories)
sp = trajectories['sparkling']
plt.figure()
for idx in range(20):
    plt.scatter(sp[idx*1536*2:(idx+1)*1536*2-1,0],sp[idx*1536*2:(idx+1)*1536*2-1,1],s=1)
plt.show()

fourier_sparkling = NonCartesianFFT(
    trajectories['sparkling'],
    shape=full_k.shape[1:],
    n_coils=full_k.shape[0],
    implementation='cpu',
    density_comp=trajectories['sparkling_w'],
)
