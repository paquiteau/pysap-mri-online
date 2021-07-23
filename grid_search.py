#!/usr/bin/env python
import copy
import yaml
from pprint import pprint
import numpy as np
import argparse
import traceback

from modopt.opt.proximity import GroupLASSO, IdentityProx
from modopt.opt.linear import Identity
from mri.operators import FFT, OWL, WaveletN

from online.operators import ColumnFFT, LASSO
from online.generators import Column2DKspaceGenerator, DataOnlyKspaceGenerator, KspaceGenerator
from online.reconstructors import OnlineReconstructor

from project_utils import load_data, create_cartesian_metrics

from results.base import Experience, loader, ungrid, get_hash

def get_operators(kspace_data, loc, mask, fourier_type=1, max_iter=80, regularisation=None, linear=None):
    """ Create the various operators from the config file. """
    n_coils = 1 if kspace_data.ndim == 2 else kspace_data.shape[0]
    shape = kspace_data.shape[-2:]
    if fourier_type == 0:  # offline reconstruction
        kspace_generator = KspaceGenerator(full_kspace=kspace_data, mask=mask, max_iter=max_iter)
        fourier_op = FFT(shape=shape, n_coils=n_coils, mask=mask)
    elif fourier_type == 1:  # online type I reconstruction
        kspace_generator = Column2DKspaceGenerator(full_kspace=kspace_data, mask_cols=loc)
        fourier_op = FFT(shape=shape, n_coils=n_coils, mask=mask)
    elif fourier_type == 2:  # online type II reconstruction
        kspace_generator = DataOnlyKspaceGenerator(full_kspace=kspace_data, mask_cols=loc)
        fourier_op = ColumnFFT(shape=shape, n_coils=n_coils)
    else:
        raise NotImplementedError
    if linear is None:
        linear_op = Identity()
    else:
        lin_cls = linear.pop('class', None)
        if lin_cls == 'WaveletN':
            linear_op = WaveletN(n_coils=n_coils, n_jobs=4, **linear)
            linear_op.op(np.zeros_like(kspace_data))
        elif lin_cls == 'Identity':
            linear_op = Identity()
        else:
            raise NotImplementedError

    prox_op = IdentityProx()
    if regularisation is not None:
        reg_cls = regularisation.pop('class')
        if reg_cls == 'LASSO':
            prox_op = LASSO(weights=regularisation['weights'])
        if reg_cls == 'GroupLASSO':
            prox_op = GroupLASSO(weights=regularisation['weights'])
        elif reg_cls == 'OWL':
            prox_op = OWL(**regularisation, n_coils=n_coils, bands_shape=linear_op.coeffs_shape)
        elif reg_cls == 'IdentityProx':
            prox_op = IdentityProx()
            linear_op = Identity()
    return kspace_generator, fourier_op, linear_op, prox_op


DATA_DIR = 'data/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid search of the MRI online program')
    parser.add_argument('-f', dest='FORCE', action='store_true', default=False,help="recomputed past computations")
    parser.add_argument('grid_file', type=str, default='result/grid_config.yml', help='config file setting data, problem and solver configuration.')
    parser.add_argument('--dry', dest='DRY', action='store_true', default=False, help='dry mode, skip the computation, but do everything else')
    parser.add_argument('--tested',type=str, default='results/tested_config.yml', help="List of past config.")
    parser.add_argument('--failed',type=str, default='results/failed_config.yml', help="List of past, failed config.")

    args = parser.parse_args()

    with open(args.grid_file) as f:
        cfg = yaml.load(f, Loader=loader)
        setups = ungrid(cfg)
    with open(args.tested) as f:
        tested_cfg = yaml.load(f, Loader=loader)
        tested_cfg = [None] if tested_cfg is None else tested_cfg
    with open(args.failed) as f:
        failed_cfg = yaml.load(f, Loader=loader)
        failed_cfg = [None] if failed_cfg is None else tested_cfg

    hash_list = list(map(get_hash, tested_cfg + failed_cfg))
    metrics_config = dict()
    metrics_results = dict()
    for idx, setup in enumerate(setups):
        data, problem, solver = copy.deepcopy(setup['data']), copy.deepcopy(setup['problem']), copy.deepcopy(
            setup['solver'])
        e = Experience(data, problem, solver)
        print(f' == {idx}/{len(setups)}: {e.hex_hash()} == ')
        pprint(e.id())
        if not args.FORCE and e.hex_hash() in hash_list:
            print('already computed')
            continue

        if not args.DRY:
            # get data
            full_k, real_img, mask_loc, final_mask = load_data(DATA_DIR, **data)
            final_k = np.squeeze(full_k * final_mask[np.newaxis])
            # create the online problem
            ksp, fft, lin, prox = get_operators(kspace_data=final_k, loc=mask_loc, mask=final_mask, **problem)
            alg_name = solver.pop('algo')
            online_pb = OnlineReconstructor(fft, lin, regularizer_op=prox, opt=alg_name, verbose=0)
            # configure metrics
            metrics_config = create_cartesian_metrics(online_pb, real_img, final_mask, final_k, estimates=5)
            # solve for the specified problem
            try:
                results = online_pb.reconstruct(ksp, **solver, **metrics_config)
            except Exception as err:
                print(" -- Failed -- ")
                print(traceback.format_exc())
                print(err)
                e.config2file(args.failed)
                continue
            metrics_results = results.pop('metrics')
            costs = results.pop('costs')
            estimates = results.pop('x_estimates', None)
            metrics_results['cost'] = {'index': np.arange(len(costs)), 'values': costs}
            metrics_results['cost_off'] = {'index': np.arange(len(costs)),
                                           'values': (np.array(metrics_results['data_res_off']['values'])
                                                      + np.array(metrics_results['reg_res']['values']))}

            e.save(estimates, metrics_results)
        if not e.hex_hash() in hash_list:
            e.config2file(args.tested)
