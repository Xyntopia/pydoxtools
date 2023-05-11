"""
The pydoxtools.operators module defines
a set of generic pipeline operators that can
be used inside of a pipeline class definition
to create your own pipelines.
"""
import abc
import typing
from abc import ABC

from typing import Callable, Iterable, Any


class Operator(ABC):
    """Base class to build extraction logic for information extraction from
    unstructured documents and loading files

    Extractors should always be stateless! This means one should not save
    any variables in them that persist over the lifecycle over a single extraction
    operation.

    - Extractors can be "hooked" into the document pipeline by using:
        pipe, out and cache calls.
    - all parameters given in "out" can be accessed through the "x" property
      (e.g. doc.x("extraction_parameter"))

    dynamic configuration of an extractor parameters can be configured through
    "config" function which will indicate to the parent document class
    to set some input parameters to this function automatically.

    configuration parameters can be set during pipeline initialization like this:

    ```
    Pipeline(source=...).config(
        first_config_parameter=...,
        seconf_config_parameter=...
    )
    ```

    In order to know what parameters of a pipeline are configurable and what
    their default values are, call the configuration parameter like this:

    Pipeline(source=...).configuration

    If the same parameters are also set in doc.pipe the parameters are
    optional and will only be taken if explicitly set through doc.config(...).

    TODO: explain configuration parameters
    """

    # TODO:  how can we anhance the type checking for outputs?
    #        maybe turn this into a dataclass?

    def __init__(self):
        # try to keep __init__ with no arguments for Operator..
        self._in_mapping: dict[str, str] = {}
        self._out_mapping: dict[str, str] = {}
        self._cache = False  # TODO: switch to "True" by default
        self._allow_disk_cache = True
        self.__node_doc__ = ""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, typing.Any] | Any:
        pass

    def pipe(self, *args, **kwargs):
        """
        configure input parameter mappings to this function

        keys: are the actual function parameters of the extractor function
        values: are the outside function names

        """
        self._in_mapping = kwargs
        self._in_mapping.update({k: k for k in args})
        return self

    def out(self, *args, **kwargs):
        """
        configure output parameter mappings to this function

        keys: are the
        """
        # TODO: rename to "name" because this actually represents the
        #       variable names of the extractor?
        self._out_mapping = kwargs
        self._out_mapping.update({k: k for k in args})
        return self

    def cache(self, allow_disk_cache=True):
        """indicate to document that we want this extractor function to be cached"""
        self._cache = True
        self._allow_disk_cache = allow_disk_cache
        return self

    def no_cache(self):
        self._cache = False
        return self

    def docs(self, doc_str: str = ""):
        self.__node_doc__ = doc_str
        return self


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


class OperatorException(Exception):
    pass


class OperatorOutputException(OperatorException):
    pass
