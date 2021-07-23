import inspect
from collections import Hashable, Set
from .utils import flatten_dict, allowed_op

MISSING = object()


class EmptySetError(Exception):
    pass


class ExperienceSet(Set, Hashable):
    """
    A set Of Experiences, defined using  Abstract Base Classes.
    """

    __hash__ = Set._hash

    wrapped_methods = (
        "difference",
        "intersection",
        "symmetric_difference",
        "union",
        "copy",
    )

    def __new__(cls, iterable=None):
        obj = super(ExperienceSet, cls).__new__(ExperienceSet)
        obj._set = set() if iterable is None else set(iterable)
        for method_name in cls.wrapped_methods:
            setattr(obj, method_name, cls._wrap_method(method_name, obj))
        return obj

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
        s = "ExperienceSet("
        for e in self._set:
            s += str(repr(e)) + "\n"
        s += f"):{len(self)} Elements"
        return s

    def filter(self, mode="loose_and", **kwargs):
        """
        Parameters
        ----------
        kwargs: key=values pairs, `key` defines a filter in the spirit of the Django framework e.g:
        results__psnr__gt = 1 return all element of the set where self.results['psnr']  > 1
        eta = 1e-1 return all element of the set where self.eta = 1e-1
        More generally a filter key is of the form:
        attr1__attr2__[..]__op
        if op is not provided, then equality test `eq` is assumed.
        A filter condition return either True, False or None if the test could have not been perform.
        mode: str
        can be either 'or', 'and' or 'loose_and', default to 'loose_and'.
        'or': at least one filter must match,
        'and' : every condition must match, and no fail is allowed,
        'loose_and': every condition must match, but ignore if a filter return None
        Returns
        -------
            An ExperienceSet whose element verify the condition set by the filters.
        """
        if len(kwargs) == 0:
            return ExperienceSet(self)

        def _match_filter(sub, kw, val):
            """check if a subject `sub` verify the condition sub.kw = val"""
            _kw = kw.split("__")
            if _kw[-1] not in allowed_op:
                op = "eq"
            else:
                op = _kw.pop()
            while _kw:
                __kw = _kw.pop(0)
                if hasattr(sub, __kw):  # attribute/property access
                    sub = getattr(sub, __kw)
                    if inspect.ismethod(sub):
                        sub = sub()
                elif hasattr(sub, "get"):  # try dict access
                    sub2 = sub.get(__kw, MISSING)
                    if sub2 is MISSING:
                        return None
                    else:
                        sub = sub2
                else:
                    if mode != "loose_and":
                        raise KeyError(f"{sub} has no accessible attribute using {kw}")
                    else:
                        return None
            return allowed_op[op](sub, val)

        final_qs = ExperienceSet()
        for sub in ExperienceSet(self):
            val_test = False
            for kw in kwargs:
                val_test = _match_filter(sub, kw, kwargs[kw])
                if mode == "or" and val_test is True:
                    break
                if mode == "and" and val_test is not True:
                    break
                if mode == "loose_and" and val_test is False:
                    break
            if (mode == "loose_and" and val_test is None) or val_test:
                final_qs.add(sub)
        return final_qs

    def get(self, **kwargs):
        qs = self.filter(**kwargs)

        if len(qs) > 1:
            print("query not specific enought, return first matching element")
        return qs._set.pop()

    def pop(self):
        return self._set.pop()

    def get_discriminant_param(self, disc=True):
        all_key = dict()
        all_key_cnt = dict()
        for exp in self:
            conf = flatten_dict(exp.__dict__)
            for k, v in conf.items():
                if k in all_key:
                    try:
                        if v == all_key[k]:
                            all_key_cnt[k] += 1
                    except ValueError as e:
                        print(e)
                        continue
                else:
                    all_key[k] = v
                    all_key_cnt[k] = 1
        if disc:
            return {k: c for k, c in all_key_cnt.items() if c < len(self)}
        else:
            return {k: all_key[k] for k, c in all_key_cnt.items() if c == len(self)}
