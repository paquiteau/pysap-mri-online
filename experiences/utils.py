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
    """ return a string with format key=value,
       and also applying formating on number if needed.
    """
    st = ""
    for k, v in kwargs.items():
        if v is None:
            continue
        if type(v) is str:
            st += v
        else:
            st += f'{k}='
            if isinstance(v,float) or isinstance(v, np.floating):
                if abs(v) > 1000 or abs(v) < 1e-2:
                    st += f'{v:.2e}'
                else:
                    st += f'{v:.2f}'
            elif type(v) is int:
                st += f'{v}'
            elif hasattr(v, '__str__'):
                st += str(v)
        st += separator
    if separator:
        st = st[:-len(separator)]
    return st