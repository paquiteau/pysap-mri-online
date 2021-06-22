from collections import defaultdict
from itertools import product
import re
import os
import sys
import copy
import json
import yaml
from pprint import pprint
import numpy as np


hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

from modopt.opt.proximity import GroupLASSO, IdentityProx
from modopt.opt.linear import Identity
from mri.operators import FFT, OWL, WaveletN

from online.operators import ColumnFFT
from online.generators import Column2DKspaceGenerator, DataOnlyKspaceGenerator, KspaceGenerator
from online.reconstructors import OnlineReconstructor

from experiences import BaseExperience

from utils import load_data, create_cartesian_metrics

# https://stackoverflow.com/a/30462009/16019838
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
       [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
      |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
      |\\.[0-9_]+(?:[eE][-+][0-9]+)?
      |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
      |[-+]?\\.(?:inf|Inf|INF)
      |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def listify(dic):
    """recursively transform value of dict into [value]"""
    for kw, v in dic.items():
        if isinstance(v, (dict, defaultdict)):
            dic[kw] = listify(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], (dict, defaultdict)):
                    v[i] = listify(v[i])
        else:
            dic[kw] = [v]
    return dic


def develop2(dic):
    """ develop the config dict to generate all possible combinaisons """
    for k, v in dic.items():
        a = []
        for vv in v:
            if isinstance(vv, (dict, defaultdict)):
                a += develop2(vv)
            if a:
                dic[k] = a
    return [dict(zip(dic.keys(), items)) for items in product(*(dic.values()))]


def get_hash(config):
    return hash(json.dumps(config, sort_keys=True))


def get_operators(kspace_data, mask_loc, mask, fourier_type=1, regularisation=None, linear=None):
    n_coils = 1 if kspace_data.ndim == 2 else kspace_data.shape[0]
    shape = kspace_data.shape[-2:]
    if fourier_type == 0:  # offline reconstruction
        kspace_generator = KspaceGenerator(full_kspace=kspace_data, mask=mask, max_iter=len(mask_loc))
        fourier_op = FFT(shape=shape, n_coils=n_coils, mask=mask)
    elif fourier_type == 1:  # online type I reconstruction
        kspace_generator = Column2DKspaceGenerator(full_kspace=kspace_data, mask_cols=mask_loc)
        fourier_op = FFT(shape=shape, n_coils=n_coils, mask=mask)
    elif fourier_type == 2:  # online type II reconstruction
        kspace_generator = DataOnlyKspaceGenerator(full_kspace=kspace_data, mask_cols=mask_loc)
        fourier_op = ColumnFFT(shape=shape, n_coils=n_coils)
    else:
        raise NotImplementedError
    if linear is None:
        linear_op = Identity()
    else:
        lin_cls = linear.pop('class', None)
        if lin_cls == 'WaveletN':
            linear_op = WaveletN(n_coils=n_coils, **linear)
            linear_op.op(np.zeros_like(kspace_data))
        else:
            raise NotImplementedError

    if regularisation is None:
        prox_op = IdentityProx()
    else:
        reg_cls = regularisation.pop('class')
        if reg_cls == 'GroupLASSO':
            prox_op = GroupLASSO(weights=regularisation['weights'])
        elif reg_cls == 'OWL':
            prox_op = OWL(**regularisation, n_coils=n_coils, bands_shape=linear_op.coeffs_shape)
        elif reg_cls == 'IdentityProx':
            prox_op = IdentityProx()
            linear_op = Identity()
            raise NotImplementedError
    return kspace_generator, fourier_op, linear_op, prox_op


class Experience(BaseExperience):
    save_folder = "results/simuls"

    def __init__(self, data, problem, solver):
        self.data = copy. deepcopy(data)
        self.problem = copy.deepcopy(problem)
        self.solver = copy.deepcopy(solver)
        super(BaseExperience).__init__()

    def id(self):
        return json.dumps(dict(data=self.data, problem=self.problem, solver=self.solver), sort_keys=True)

    def config2file(self, file, append=True):
        mode = 'a' if append else 'w'
        with open(file, mode) as f:
            f.write(yaml.dump([dict(data=self.data, problem=self.problem, solver=self.solver)]))


DATA_DIR = 'data/'
FORCE_COMPUTE = False
DRY_MODE = False
if __name__ == '__main__':

    filename = 'results/grid_config.yml'
    with open(filename) as f:
        cfg = yaml.load(f, Loader=loader)
        setups = develop2(listify(cfg))
    with open('results/tested_config.yml') as f:
        tested_cfg = yaml.load(f, Loader=loader)
        tested_cfg = [None] if tested_cfg is None else tested_cfg
    hash_list = list(map(get_hash, tested_cfg))
    pprint(json.dumps(tested_cfg[0],sort_keys=True))
    print(hash_list)
    metrics_config = dict()
    metrics_results = dict()
    for idx, setup in enumerate(setups):
        data, problem, solver = copy.deepcopy(setup['data']), copy.deepcopy(setup['problem']), copy.deepcopy(setup['solver'])
        e = Experience(data, problem, solver)
        print(f' == {idx}/{len(setups)}: {hash(e.id)} == ')
        pprint(e.id())
        if not FORCE_COMPUTE and hash(e) in hash_list:
            print('already computed')
            continue

        if not DRY_MODE:
            # get data
            full_k, real_img, mask_loc, final_mask = load_data(DATA_DIR, **data)
            final_k = np.squeeze(full_k * final_mask[np.newaxis])
            # create the online problem
            ksp, fft, lin, prox = get_operators(kspace_data=final_k, mask_loc=mask_loc, mask=final_mask, **problem)
            alg_name = solver.pop('algo')
            online_pb = OnlineReconstructor(fft, lin, regularizer_op=prox, opt=alg_name, verbose=0)
            # configure metrics
            metrics_config = create_cartesian_metrics(online_pb, real_img, final_mask, final_k)
            # solve for the specified problem
            try:
                xf, costs, metrics_results = online_pb.reconstruct(ksp, **solver, **metrics_config)
            except:
                print(" -- Failed -- ")
                continue
            metrics_results['cost'] = {'index': np.arange(len(costs)), 'values': costs}
            e.save_results(metrics_results)
        e.config2file('results/tested_config.yml')