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
print('import experiences.experience')

import os
import pickle
import functools
from .set import ExperienceSet
from .results import Results

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
        del self._theset
        self._theset = ExperienceSet()

    def create(self, *args, **kwargs):
        return BaseExperience(*args, **kwargs)

    def set_attributes(self, an_object):
        self._object_attributes = [attr_name for attr_name in an_object.__dict__.keys()]

    def all(self):
        return self._theset

    def filter(self, **kwargs):
        return self._theset.filter(**kwargs)

    def filter_plot(self, *args, **kwargs):
        return self._theset.filter_plot(*args, **kwargs)


class BaseExperience:
    objects = ExperienceManager()
    save_folder = ''

    def __init__(self):
        # add to the Manager:
        if not len(self.objects):
            self.objects.set_attributes(self)
        self.objects.add(self)

    def save(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def metrics_file(self):
        return f'{self.save_folder}/{hash(self)}.pkl'

    @property
    def data_file(self):
        return f'{self.save_folder}/x_{hash(self)}.pkl'

    @functools.cached_property
    def results(self):
        with open(self.metrics_file, 'rb') as f:
            res = pickle.load(f)
        return res

    @functools.cached_property
    def xf(self):
        with open(self.data_file, 'rb') as f:
            res = pickle.load(f)
        return res

    def id(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.id())

    def __repr__(self):
        return self.id
