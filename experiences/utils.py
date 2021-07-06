from collections import MutableMapping


import numpy as np


allowed_op = {
    'eq': lambda x, y: x == y,
    'neq': lambda x, y: x != y,
    'le': lambda x, y: x <= y,
    'lt': lambda x, y: x < y,
    'gt': lambda x, y: x > y,
    'ge': lambda x, y: x >= y,
    'in': lambda x, y: x in y,
}


def flatten_dict(d, parent_key='', sep='__'):
    """ Flatten a dictionnary """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    ret = dict(items)
    ret.pop('results', None)
    return ret


def key_val(separator=', ', **kwargs):
    """ Transform a dict of key: value into a string with format key=value,
       and also applying formating on number if needed.
    """
    st = ""
    for k, v in kwargs.items():
        k = k.split('__')[-1]
        if v is None:
            continue
        if type(v) is str:
            st += v
        else:
            if isinstance(v, float) or isinstance(v, np.floating):
                if abs(v) > 1000 or abs(v) < 1e-2:
                    st += f'{k}={v:.2e}'
                else:
                    st += f'{k}={v:.2f}'
            elif type(v) is int:
                st += f'{k}={v}'
            elif type(v) is bool:
                if v:
                    st += f'{k}'
                else:
                    st += f'not {k}'
            else:
                continue
        st += separator
    if separator:
        st = st[:-len(separator)]
    return st



