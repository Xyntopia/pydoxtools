"""
The pydoxtools.operators module defines
a set of generic pipeline operators that can
be used inside of a pipeline class definition
to create your own pipelines.
"""

from typing import Callable, Iterable

from pydoxtools.document_base import Operator


class Alias(Operator):
    """Connect extractor variables with Aliases"""

    def __init__(self, **kwargs):
        super().__init__()
        self.pipe(**{v: v for k, v in kwargs.items()})
        self.out(**{v: k for k, v in kwargs.items()})

    def __call__(self, **kwargs):
        return kwargs


class Constant(Operator):
    """declare one ore more constant values"""

    def __init__(self, **kwargs):
        super().__init__()
        self.const_result = kwargs
        self.out(**{k: k for k in kwargs})
        self.no_cache()

    def __call__(self):
        return self.const_result


class LambdaOperator(Operator):
    """Wrap an arbitrary function as an Operator"""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class ElementWiseOperator(Operator):
    """
    Take a function and apply it elementwise to
    an iterable. Return a list or iterator.

    the "elements" argument will be evaluated
    element-wise. You can specify additional arguments for the
    function using *args and **kwargs.
    """

    # TODO: automatically add documentation from self._func
    def __init__(self, func: Callable, return_iterator: bool):
        super().__init__()
        self._func = func
        self._return_iterator = return_iterator

    def __call__(self, elements: Iterable, *args, **kwargs):
        res = (self._func(element, *args, **kwargs) for element in elements)
        if self._return_iterator:
            return res
        else:
            return list(res)


class DictSelector(Operator):
    def __call__(self, selectable: dict) -> Callable[[...], dict]:
        def selector(*args, **kwargs) -> dict:
            selection = {a: selectable.get(a, None) for a in args}
            selection.update({v: selectable.get(k, None) for k, v in kwargs.items()})
            return selection

        return selector


class DataMerger(Operator):
    """
    Merges data from several sources into a single dictionary,
    it will try to convert everything into strings!
    """

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        out = kwargs
        return {"joint_data": out}
