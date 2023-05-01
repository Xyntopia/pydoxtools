"""
The pydoxtools.operators module defines
a set of generic pipeline operators that can
be used inside of a pipeline class definition
to create your own pipelines.
"""

from typing import Callable, Iterable

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


class Configuration(Operator):
    """
    This is a special operator which can be used to configure a pipeline.

    Declare some configuration values which can then be used as inputs for
    other operators.

    It takes a list of key-value pairs where the key
    is the target variable name and the value is the
    standard configuration value.

    All "Configuration" values can be changed through the "config"
    function in a pipeline.

    When using a Configuration, we do not need an "out" mapping, as it will
    directly be mapped on the configuration keys. We can optionally do this though.

    The Question & Answering part of the pydoxtools.Document class
    was specified with this config function like this:

        QamExtractor(model_id=settings.PDXT_STANDARD_QAM_MODEL)
            .pipe(text="full_text").out("answers").cache().config(trf_model_id="qam_model_id"),

    In this case, when calling a document we can dynamically configure the
    pipeline with the "qam_model_id" parameter:

        doc = Document(
            fobj=doc_str, document_type=".pdf"
        ).config(dict(qam_model_id='deepset/roberta-base-squad2'))


    """

    def __init__(self, **configuration_map):
        super().__init__()
        self._configuration_map = configuration_map
        self.no_cache()  # we don't need this, as everything is already saved in the _configuration_map
        # use the configuration map directly as output mapping
        self.out(*list(configuration_map.keys()))

    def __call__(self):
        return self._configuration_map


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
                    try:
                        props = doc.property_dict(*properties)
                    except OperatorException:
                        # we just continue  if an error happened. This is why we are "forgiving"
                        if len(properties) > 1:
                            props = {"Error": "OperatorException"}
                        else:
                            continue

                    if len(properties) == 1:
                        yield props[properties[0]]
                    else:
                        yield props

            return safe_extract
