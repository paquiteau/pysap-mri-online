import os
import sys
import pickle
import re
import copy
import yaml
import json
import inspect
import hashlib
import numpy as np
from collections import defaultdict
import warnings
from itertools import product
import matplotlib.pyplot as plt

from experiences.experience import BaseExperience

from experiences.results import Results
from experiences.utils import key_val

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


class Experience(BaseExperience):
    save_folder = "results/simuls"

    def __init__(self, data, problem, solver):
        self.data = copy.deepcopy(data)
        self.problem = copy.deepcopy(problem)
        self.solver = copy.deepcopy(solver)
        super(BaseExperience).__init__()
        
    def __repr__(self):
        return str(self.id())

    def save(self, xf, results_dict):
        with open(f'{self.metrics_file}', 'wb') as f:
            pickle.dump(Results(**results_dict), f)
        if xf:
            with open(f'{self.data_file}', 'wb') as f:
                pickle.dump(xf, f)
        else:
            self.xf = None

    def id(self):
        return json.dumps(dict(data=self.data, problem=self.problem, solver=self.solver), sort_keys=True)

    def config2file(self, file, append=True):
        mode = 'a' if append else 'w'
        with open(file, mode) as f:
            f.write(yaml.dump([dict(data=self.data, problem=self.problem, solver=self.solver)]))


def ungrid(config_gen):
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
        """ develop the config dict to generate all possible combinations """
        for k, v in dic.items():
            a = []
            for vv in v:
                if isinstance(vv, (dict, defaultdict)):
                    a += develop2(vv)
                if a:
                    dic[k] = a
        return [dict(zip(dic.keys(), items)) for items in product(*(dic.values()))]

    if isinstance(config_gen, list):
        setups = []
        for cf in config_gen:
            setups += develop2(listify(cf))
    else:
        setups = develop2(listify(config_gen))
    return setups

def get_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


