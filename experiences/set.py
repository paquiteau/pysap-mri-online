import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inspect
from collections import Hashable, Set

from .utils import key_val, flatten_dict, allowed_op

MISSING = object()

class EmptySetError(Exception):
    pass

class ExperienceSet(Set, Hashable):
    """
    A set Of Experiences, defined using  Abstract Base Classes.
    """
    __hash__ = Set._hash

    wrapped_methods = ('difference',
                       'intersection',
                       'symetric_difference',
                       'union',
                       'copy')

    def __new__(cls, iterable=None):
        selfobj = super(ExperienceSet, cls).__new__(ExperienceSet)
        selfobj._set = set() if iterable is None else set(iterable)
        for method_name in cls.wrapped_methods:
            setattr(selfobj, method_name, cls._wrap_method(method_name, selfobj))
        return selfobj

    @classmethod
    def _wrap_method(cls, method_name, obj):
        def method(*args, **kwargs):
            result = getattr(obj._set, method_name)(*args, **kwargs)
            return ExperienceSet(result)

        return method

    def __getattr__(self, attr):
        """Make sure that we get things like issuperset() that aren't provided
        by the mix-in, but don't need to return a new set."""
        return getattr(self._set, attr)

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._set)

    def __repr__(self):
        s = 'ExperienceSet('
        for e in self:
            s += repr(e) + '\n'
        s += f'):{len(self)} Elements'
        return s

    def filter(self, **kwargs):
        mode = kwargs.pop('mode', 'loose_and')
        if len(kwargs) == 0:
            return ExperienceSet(self)

        def match_filter(sub, kw, val):
            _kw = kw.split('__')
            if _kw[-1] not in allowed_op:
                op = 'eq'
            else:
                op = _kw.pop()
            while _kw:
                __kw = _kw.pop(0)
                if hasattr(sub, __kw):
                    sub = getattr(sub, __kw)
                    if inspect.ismethod(sub):
                        sub = sub()
                elif hasattr(sub, 'get'):  # try dict access
                    sub2 = sub.get(__kw, MISSING)
                    if sub2 is MISSING:
                        return None
                    else:
                        sub = sub2
                else:
                    raise KeyError(f'{sub} has no accessible attribute using {kw}')

            return allowed_op[op](sub, val)

        final_qs = ExperienceSet()
        for sub in ExperienceSet(self):
            val_test = False
            for kw in kwargs:
                val_test = match_filter(sub, kw, kwargs[kw])
                if mode == 'or' and val_test is True:
                    break
                if mode == 'and' and val_test is not True:
                    break
                if mode == 'loose_and' and val_test is False:
                    break
            if (mode == 'loose_and' and val_test is None) or val_test:
                final_qs.add(sub)
        return final_qs

    def _discrimininant_param(self, disc=True):
        all_key = dict()
        all_key_cnt = dict()
        for exp in self:
            conf = flatten_dict(exp.__dict__)
            for k, v in conf.items():
                if k in all_key:
                    if v == all_key[k]:
                        all_key_cnt[k] += 1
                else:
                    all_key[k] = v
                    all_key_cnt[k] = 1
        if disc:
            return {k for k, c in all_key_cnt.items() if c < len(self)}
        else:
            return {k: all_key[k] for k, c in all_key_cnt.items() if c == len(self)}

    def to_dataframe(self, attr):
        key_of_interest = self._discrimininant_param()
        data = dict()
        for exp in self:
            legend_key = dict()
            for k in key_of_interest:
                if '__' in k:
                    k1, k2 = k.split('__')
                    legend_key[k2] = getattr(exp, k1).get(k2, None)
                else:
                    legend_key[k] = getattr(exp, k)
            col_name = key_val(**legend_key)
            data[col_name] = getattr(exp.results, attr)
        return pd.DataFrame(data)

    def filter_plot(self, *attrs, log=False, **kwargs):
        qs = self.filter(**kwargs)
        title = key_val(**qs._discrimininant_param(disc=False))
        if not qs:
            raise EmptySetError(f'No Element matching {kwargs}')
        for attr in attrs:
            plt.figure()
            df = qs.to_dataframe(attr=attr)
            ax = df.plot(logy=log,
                         title=title,
                         ylabel=attr,
                         xlabel='index',
                         sort_columns=True)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        return qs

    def has_good_behavior(self, metric, decrease=True):
        qs = ExperienceSet()
        for exp in self:
            if exp.results.good_behavior(metric, decrease=decrease):
                qs.add(exp)
        return qs
