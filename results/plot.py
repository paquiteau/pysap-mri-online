#!/usr/bin/env python3

import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from array2gif import write_gif
from IPython.display import display, Image
from .base import key_val

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
    'blue',
    'orange',
    'green',
    'red',
    'purple',
    'brown',
    'purple',
    'pink',
    'gray',
    'black',
    'yellow',
]

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
    if len(ignored) ==0:
        return disc
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


def plot_metrics(dataset, metrics=None, ignore_keys=None, default_offset=(80,0,1),log=True, title=False, unique=True):
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
    unique_list=[]
    fig, axs = plt.subplots(len(metrics), 1,facecolor='w')
    if len(metrics) == 1:
        axs = (axs,)
    style = ('dashdot', 'dashed', 'solid')
    replace_strings=[('fourier_type','type'),
                     (', beta',', $\\beta$'),
                     (', eta',', $\\eta$'),
                     ('reg_factor','reg'),
                     ('nb_run','run')
                    ]
    
    
    linewidth = (2, 1)
    # retrieve only the usefull discriminant keys
    disc_clean = filter_disc(dataset.get_discriminant_param(), *ignore_keys)
    disc_clean_colors = filter_disc(disc_clean, 'fourier_type')
    # create the colors list
    n_colors=1
    for v in disc_clean.values():
        n_colors *=v
    colors = []    
    for _ in range(int(n_colors / len(DEFAULT_COLORS)) + 1):
        colors += copy.deepcopy(DEFAULT_COLORS)
    
    color_hash_table = dict()
    # plot each experience
    for idx, exp in enumerate(dataset):
        ls = style[exp.problem['fourier_type']]
        offset = default_offset[exp.problem['fourier_type']]
        cl = get_color_list(exp, disc_clean_colors, colors, color_hash_table)
       # lw = linewidth[exp.solver.get('nb_run', 1) - 1]
        lw = 1
        if unique:
            id = exp.id()
            if id in unique_list:
                print(id)
                continue
            else:
                unique_list.append(id)
        lbl = get_label(exp, disc_clean)
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
            if default_offset:
                ax.axvline(x=default_offset[0],linestyle='solid',color='gray',linewidth=0.2)
    # legend clean_up
    if len(dataset)>1:
        handles, labels = axs[0].get_legend_handles_labels()
        unique_labels=[]
        unique_handles=[]
        for i in range(len(labels)):
            for x,y in replace_strings:
                labels[i] = labels[i].replace(x,y)
            if labels[i] not in unique_labels:
                unique_labels.append(labels[i])
                unique_handles.append(handles[i])
                
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(unique_labels, unique_handles), key=lambda t: t[0]))
        
        axs[0].legend(handles, labels, loc="upper left",bbox_to_anchor=(1.01,1.0),fontsize=6)
    if title:
         fig.suptitle(key_val(**dataset.get_discriminant_param(disc=False)))
    else:
        print(key_val(**dataset.get_discriminant_param(disc=False)))
    fig.tight_layout()
    return fig

def plot_line(dataset, ignore_keys=None, default_offset=0,log=False):
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
    fig, ax = plt.subplots(1, 1,facecolor='w')
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
       
        data = abs(exp.xf[-1][320])
        if log:
            ax.semilogy(np.arange(len(data)) + offset, data,
                    linestyle=ls, color=cl, linewidth=lw,
                    label=lbl)
        else:
            ax.plot(np.arange(len(data)) + offset, data,
                    linestyle=ls, color=cl, linewidth=lw,
                    label=lbl)
    ax.set_ylabel('value')
    if len(dataset)>1:
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        fig.suptitle(key_val(**dataset.get_discriminant_param(disc=False)))
        ax.legend(handles, labels, loc="upper left",bbox_to_anchor=(1.01,1.0))
    fig.tight_layout()
    return fig
def make_gif(exp, **kwargs):
    """
    Create a gif from the estimates
    """
    estimates = exp.xf
    filename = f"{exp.save_folder}/{hash(exp)}.gif"
    write_gif(estimates, filename, **kwargs)
    Image(filename)
    
    