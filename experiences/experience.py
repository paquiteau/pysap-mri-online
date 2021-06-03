"""
this module provide a high level management of testing configuration and results.
It is highly inspired by the Django Framework.

# create Experiences
for c in config:
    ex = Experience(**c)
    results = get_results(c)
    Experience.save_results(results)
######
for c in config:
    ex = Experience(**c)

Experience.objects.filter(and=True, 'alg="condat"', 'reg_kwargs["weights"]>1e-5')

"""

import os
import pickle
import functools
import hashlib

from .set import ExperienceSet
from .results import Results
MISSING = object()


class ExperienceManager:
    """
    Manager instantiate once as class attribute of ExperienceRealisation.
    Creates Set of Experiences and filter them.
    """
    def __init__(self):
        # this will hold a list of all attributes from your custom class, once
        # initiated
        self._object_attributes = None
        self._theset = ExperienceSet()

    def add(self, item):
        self._theset.add(item)

    def discard(self, item):
        self._theset.discard(item)

    def __iter__(self):
        return iter(self._theset)

    def __len__(self):
        return len(self._theset)

    def __contains__(self, item):
        try:
            return item in self._theset
        except AttributeError:
            return False

    def flush(self):
        self._theset = ExperienceSet()

    def create(self, *args, **kwargs):
        return ExperienceRealization(*args, **kwargs)

    def set_attributes(self, an_object):
        self._object_attributes = [attr_name for attr_name in an_object.__dict__.keys()]

    def all(self):
        return self._theset

    def filter(self, **kwargs):
        return self._theset.filter(**kwargs)

    def filter_plot(self, *args, **kwargs):
        return self._theset.filter_plot(*args, **kwargs)


class ExperienceRealization:
    objects = ExperienceManager()
    save_folder = ''

    def __init__(self, online: int, linear_cls='',
                 reg_cls='', reg_kwargs=None, opt='',
                 alg_kwargs=None):
        self.online = online
        self.linear_cls = linear_cls
        self.reg_cls = reg_cls
        self.reg_kwargs = reg_kwargs
        self.opt = opt
        self.alg_kwargs = alg_kwargs

        # add to the Manager:
        if not len(self.objects):
            self.objects.set_attributes(self)
        self.objects.add(self)

    def save_results(self, results_dict):
        with open(f'{self.save_folder}/{hash(self)}.pkl', 'wb') as f:
            pickle.dump(Results(**results_dict), f)

    @property
    def save_file(self):
        return f'{self.save_folder}/{hash(self)}.pkl'

    @functools.cached_property
    def results(self):
        with open(self.save_file, 'rb') as f:
            res = pickle.load(f)
        return res

    @property
    def has_saved_results(self):
        return os.path.exists(self.save_file)

    @property
    def st(self):
        st = str(self.online) + self.linear_cls \
             + self.reg_cls + repr(self.reg_kwargs) \
             + self.opt + repr(self.alg_kwargs)
        return st

    def __hash__(self):
        return int(hashlib.md5(self.st.encode()).hexdigest(), 16)

    def __repr__(self):
        return self.st