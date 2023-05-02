"""
The pydoxtools.operators module defines
a set of generic pipeline operators that can
be used inside of a pipeline class definition
to create your own pipelines.
"""

from typing import Callable, Iterable, Any

from pydoxtools.document_base import Operator, OperatorException, Pipeline
from pydoxtools.list_utils import iterablefyer


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


def forgiving_extract(doc: Pipeline, properties: list[str]) -> dict[str, Any]:
    try:
        props = doc.property_dict(*properties)
        return props
    except OperatorException:
        # we just continue  if an error happened. This is why we are "forgiving"
        return {"Error": "OperatorException"}


class ForgivingExtractIterator(Operator):
    """
    Creates a loop to extract properties from a list of
    pydoxtools.Document.

    method = "list":
        Iterator can be configured to output either a list property dictionaries

    method = "single_property"
        output of the iterator will be a single, flat list of said property
    """

    def __init__(self, method="list"):
        super().__init__()
        self._method = method

    def __call__(self, doc_list: list[Pipeline]) -> Callable:
        """Define a safe_extract iterator because we want to stay
        flexible here and not put all this data in our memory...."""

        # TODO: define a more "static" propertylist function which
        #       could be configured on documentset instantiation
        if self._method == "list":
            def safe_extract(properties: list[str] | str) -> list[dict]:
                properties = iterablefyer(properties)
                for doc in doc_list:
                    props = forgiving_extract(doc, properties)

                    if len(properties) == 1:
                        yield props[properties[0]]
                    else:
                        yield props

            return safe_extract
