import os
import sys
import pickle
import re
import copy
import yaml
import json
import inspect
import numpy as np
from collections import defaultdict
import warnings
from itertools import product
import matplotlib.pyplot as plt

from experiences.experience import BaseExperience

from experiences.results import Results
from experiences.utils import key_val

DEFAULT_COLORS=[
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

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
        with open(f'{self.save_folder}/{hash(self)}.pkl', 'wb') as f:
            pickle.dump(Results(**results_dict), f)
        if xf:
            with open(f'{self.save_folder}/x_{hash(self)}.pkl', 'wb') as f:
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

def set_hashseed(seed=0):
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)


def get_hash(config):
    return hash(json.dumps(config, sort_keys=True))


def plot_metrics(dataset, metrics=None, ignore_keys=None, default_offset=80,log=True):
    """
    Plot metrics of dataset, and take care of the styling:
    the linestyle is determined by the fourier type
    the linewidth is determined by the number of run performed
    the color is determined by the rest of relevant parameters, which are not in ignored keys.
    Parameters
    ----------
    dataset: ExperienceSet object
    metrics: list|tuple of metrics to plot.
    ignore_keys: list|tuple of keys to ignore in the crafting of the legend labels
    default_offset: int , offset on the x axis (time) for the offline reconstruction
    Returns
    -------
    Figure handle
    """
    if ignore_keys is None:
        ignore_keys = []
    if metrics is None:
        metrics = ['psnr']

    def get_label(e, disc_keys):
        """
        Generate a legend label for an experience, from its discriminants properties
        Parameters
        ----------
        e: Experience object, containing data and parameters
        disc_keys: dict of discriminant keys
        Returns
        -------
        string formatted in the key=value format.
        """
        legend_key = dict()
        for key in disc_keys:
            _key = key.split('__')
            sub = copy.copy(e)
            while _key:
                __key = _key.pop(0)
                if hasattr(sub, __key):  # attribute/property access
                    sub = getattr(sub, __key)
                    if inspect.ismethod(sub):
                        sub = sub()
                elif hasattr(sub, 'get'):  # try dict access
                    sub = sub.get(__key)
            legend_key[__key] = sub
        return key_val(**legend_key)

    def filter_disc(disc, *ignored):
        clean = dict()
        for dkey in disc:
            is_ignored = False
            for key in ignored:
                if key in dkey:
                    is_ignored = True
                    break
            if is_ignored:
                continue
            clean[dkey] = disc[dkey]
        return clean

    def get_color_list(e, disc, colors, hash_table):
        label = hash(get_label(e, disc))
        if label not in hash_table:
            hash_table[label] = colors.pop(0)
        return hash_table[label]

    fig, axs = plt.subplots(len(metrics), 1,facecolor='w')
    if len(metrics) == 1:
        axs = (axs,)
    style = ('dotted', 'dashed', 'solid')
    linewidth = (2, 1)
    # retrieve only the usefull discriminant keys
    disc_clean = filter_disc(dataset.get_discriminant_param(), *ignore_keys)
    disc_clean_colors = filter_disc(disc_clean, 'fourier_type')
    # create the colors list
    colors = copy.deepcopy(DEFAULT_COLORS)
    n_colors = sum(disc_clean.values())
    if n_colors > len(colors):
      #  warnings.warn("The number of different colors is not big enough")
        colors *= int(n_colors / len(colors)) + 1
    color_hash_table = dict()
    # plot each experience
    for idx, exp in enumerate(dataset):
        ls = style[exp.problem['fourier_type']]
        offset = default_offset if exp.problem['fourier_type'] == 0 else 0
        cl = get_color_list(exp, disc_clean_colors, colors, color_hash_table)
        lw = linewidth[exp.solver.get('nb_run', 1) - 1]
        if exp.solver.get('nb_run', 1) == 1:
            lbl = get_label(exp, disc_clean)
        else:
            lbl = None
        for ax, metric in zip(axs, metrics):
            data = getattr(exp.results, metric)
            if log:
                ax.semilogy(np.arange(len(data)) + offset, data,
                        linestyle=ls, color=cl, linewidth=lw,
                        label=lbl)
            else:
                ax.plot(np.arange(len(data)) + offset, data,
                        linestyle=ls, color=cl, linewidth=lw,
                        label=lbl)
            ax.set_ylabel(metric)

    handles, labels = axs[0].get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    fig.suptitle(key_val(**dataset.get_discriminant_param(disc=False)))
    axs[0].legend(handles, labels, loc="upper left",bbox_to_anchor=(1.01,1.0))
    return fig
