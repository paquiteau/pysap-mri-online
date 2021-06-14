import os
import time
import itertools
import copy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from mri.operators import FFT, WaveletN, OWL
from modopt.opt.proximity import GroupLASSO, IdentityProx

from online.generators.column import Column2DKspaceGenerator, DataOnlyKspaceGenerator
from online.reconstructors.reconstructor import OnlineReconstructor
from online.operators.fourier import ColumnFFT
from online.metrics import crop_center_square, ssos

from utils import implot, load_data, create_cartesian_metrics
from experiences import ExperienceRealization


def product_dict(**kwargs):
    for keys in kwargs.keys():
        if type(kwargs[keys]) not in [list, tuple, np.ndarray]:
            kwargs[keys] = [kwargs[keys]]
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


DATA_DIR = "/home/pac/gits/phd/mri-online/data/"
RESULT_DIR = "../data/results/"
N_JOBS = 1
results = dict()

full_k, real_img, mask_loc, final_mask = load_data(DATA_DIR, 1)
real_img = real_img.astype(np.complex128)
# use mono-coil data
full_k = sp.fft.ifftshift(sp.fft.fft2(real_img, s=(320, 320), norm="ortho"))
final_mask = crop_center_square(final_mask)
final_k = full_k * final_mask

kspace_gen = Column2DKspaceGenerator(final_k, mask_cols=mask_loc)
kspace_gen2 = DataOnlyKspaceGenerator(final_k, mask_cols=mask_loc)

K_DIM = kspace_gen.shape[-2:]
N_COILS = kspace_gen.shape[0] if full_k.ndim == 3 else 1

fourier_op = FFT(shape=K_DIM, n_coils=N_COILS, mask=final_mask)
fourier_op2 = ColumnFFT(shape=K_DIM, n_coils=N_COILS)

linear_op = WaveletN("sym8", nb_scale=4, n_coils=N_COILS, n_jobs=N_JOBS)
linear_op.op(np.zeros_like(final_k))

fourier = [(kspace_gen2, fourier_op2, 2)]

regs = [
    # {
    #     'class': GroupLASSO,
    #     'kwargs': {'weights': [5e-4]},
    #     'cst_kwargs': {}
    # },
    {
        "class": IdentityProx,
        "cst_kwargs": {},
        "kwargs": {},
    }
]

algo = [
    # {
    #     'opt': 'condatvu',
    #     'kwargs': {},
    # },
    # {
    #     'opt': 'pogm',
    #     'kwargs': {}
    # },
    # {
    #     'opt': 'vanilla',
    #     'kwargs': {
    #         'eta': [0.1, 0.5, 1.0]
    #     }
    # },
    # {
    #     'opt': 'adagrad',
    #     'kwargs': {
    #         'eta': [0.1, 0.5, 1.0]
    #     }
    # },
    {
        "opt": "rmsprop",
        "kwargs": {
            "eta": [0.001, 0.01, 0.1],
            "gamma": [0.0],
        },
    },
    # {
    #     'opt': 'momentum',
    #     'kwargs': {
    #         'eta': 1.0,
    #         'beta': [0.3, 0.4, 0.5, 0.6],
    #     }
    # },
    # {
    #     'opt': 'adam',
    #     'kwargs': {
    #         'eta': [0.0001,0.001],
    #         'beta': [0.001],
    #         'gamma': [0.001],
    #
    #     }
    # },
]
FORCE_COMPUTE = False
count = 0
configs = list()
for f_config in fourier:
    for reg_config in regs:
        for reg_param in product_dict(**reg_config["kwargs"]):
            for alg_config in algo:
                for alg_param in product_dict(**alg_config["kwargs"]):
                    count += 1
                    config = {
                        "fourier_op": f_config[1],
                        "kspace_gen": f_config[0],
                        "linear_op": linear_op,
                        "reg_class": reg_config["class"],
                        "reg_kwargs": reg_param,
                        "opt": alg_config["opt"],
                        "alg_kwargs": alg_param,
                    }
                    config_sig = {
                        "online": f_config[2],
                        "linear_cls": linear_op.__class__.__name__,
                        "reg_cls": reg_config["class"].__name__,
                        "reg_kwargs": reg_param,
                        "opt": alg_config["opt"],
                        "alg_kwargs": alg_param,
                    }
                    configs.append((config_sig, config))

print("total number of config:", count)
ExperienceRealization.save_folder = os.path.join(os.getcwd(), "results")
for idx, (cs, c) in enumerate(configs):
    e = ExperienceRealization.objects.create(**cs)
    print(f"========== {idx}/{count} ==========")
    print(e.st)
    if not e.has_saved_results or FORCE_COMPUTE:
        print(c["opt"])
        online_pb = OnlineReconstructor(
            fourier_op=c["fourier_op"],
            linear_op=c["linear_op"],
            regularizer_op=c["reg_class"](**c["reg_kwargs"]),
            opt=c["opt"],
            verbose=0,
        )
        metrics = create_cartesian_metrics(online_pb, real_img, final_mask, final_k)
        xf, costs, metrics_results = online_pb.reconstruct(
            c["kspace_gen"],
            cost_op_kwargs={"cost_interval": 1},
            metrics=metrics,
            metric_call_period=1,
            **c["alg_kwargs"],
        )
        struct = copy.deepcopy(metrics_results[list(metrics_results.keys())[0]])
        metrics_results["cost"] = struct
        metrics_results["cost"]["values"] = costs
        print(f"{metrics_results['data_res_off']['values'][-1]:.2e}")
        e.save_results(metrics_results)
    else:
        print("result for config found")
