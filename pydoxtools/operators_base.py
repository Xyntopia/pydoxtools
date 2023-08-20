"""
The pydoxtools.operators module defines
a set of generic pipeline operators that can
be used inside of a pipeline class definition
to create your own pipelines.
"""
from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import abc
import functools
import itertools
import typing
from abc import ABC
from typing import Callable, Iterable, Any

import pydantic

OperatorReturnType = typing.TypeVar('OperatorReturnType')


class OperatorResult(pydantic.BaseModel):
    data: typing.Any
    source: typing.Any

    class Config:
        arbitrary_types_allowed = True


class Operator(ABC, typing.Generic[OperatorReturnType]):
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

    For Operator.out and Operator.pipe mappings, the "keys" always
    refer to the parameter names *inside* the operator and the "values"
    refer to the parameter names *outside* in the pipeline and the names
    accessible to other Operators. This distinction is important
    as we might use one Operator with different configurations and want to
    avoid variable collisions in the pipeline.

    TODO: explain configuration parameters
    """

    # TODO:  how can we anhance the type checking for outputs?
    #        maybe turn this into a dataclass?

    def __init__(self):
        # try to keep __init__ with no arguments for Operator..
        self._in_mapping: dict[str, str] = {}
        self._out_mapping: dict[str, str] = {}
        self._output_type: dict[str, Any] | Any = {}
        self._cache = False  # TODO: switch to "True" by default
        self._allow_disk_cache = True
        self._default = None
        self.__node_doc__: str | dict[str, str] = self.__doc__

    def map_output_types(self, output_type):
        """
        map a list of output types to our output keys for type checking
        purposes.
        """
        if isinstance(output_type, dict):
            return {v: output_type[k] for k, v in self._out_mapping.items()}

        if not isinstance(output_type, tuple):
            # if the output isn't in a tuple
            # if we have an output type like list[int] we want it to be enclosed in a tuple
            output_type = (output_type,)
        if dict == typing.get_origin(output_type[0]) and (len(output_type) == 1):
            # extract new output type from dict
            noutput_type = (typing.get_args(output_type[0])[1],)
        else:
            noutput_type = output_type

        out_keys = self._out_mapping.values()
        if (len(noutput_type) < len(out_keys)) and len(noutput_type) == 1:
            typing_dict = dict(itertools.zip_longest(out_keys, noutput_type, fillvalue=noutput_type[0]))
        else:
            typing_dict = dict(zip(out_keys, noutput_type, strict=True))
        return typing_dict

    @functools.cached_property
    def return_type(self) -> dict[Any]:
        """
        this property returns the type that was specified for the function operator

        key: we can choose from an output dictionary, for which key we would like to have
             the return type.
        """

        # TODO: what do we do if callable is a dict?? --> incorporate that!
        #       we need to check output valus and map output value dictionary
        #       to types
        if self._output_type:
            typing_dict = self.map_output_types(self._output_type)
            return typing_dict

        try:
            if type_hints := typing.get_type_hints(self.__call__):
                output_type = type_hints.get('return', typing.Any)
                if output_type != CallableType:
                    # TODO: make all output types a type of something like "operator.result".
                    #       this is important, this way we can
                    #       break the implicit dependency here between dict types which get mapped
                    #       in the pipeline.x() function. Where a dict automatically gets mapped
                    #       to the output map
                    return self.map_output_types((output_type,))
        except TypeError:
            raise TypeError(f"Could not get type hints for {self} with output: {self._out_mapping}")

        if typed_class := getattr(self, "__orig_class__", None):
            output_type = typing.get_args(typed_class)
            return self.map_output_types(output_type)

        return {}

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> OperatorReturnType:
        pass

    def input(self, *args, **kwargs):
        """
        configure input parameter mappings to this function

        keys: are the actual function parameters of the extractor function
        values: are the outside function names
        """
        self._in_mapping = kwargs
        self._in_mapping.update({k: k for k in args})
        return self

    def t(self, output_type=None, **output_types_dict):
        """declare an optional return type which is used for documentation & downstream tasks"""
        if output_type:
            self._output_type = output_types_dict
        else:
            self._output_type = output_types_dict
        return self

    def out(self, *args, **kwargs):
        """
        configure output parameter mappings to this function

        every string in *args will map the output from this operator to
        a variable with the same name in the pipeline.

        every mapping in **kwargs will map the "key" of the output of
        the operator to the "value" inside the pipeline
        """
        # TODO: rename to "name" because this actually represents the
        #       variable names of the extractor?
        self._out_mapping = kwargs
        self._out_mapping.update({k: k for k in args})
        return self

    @property
    def multiple_outputs(self):
        """indicate that this operator has multiple outputs.
        If it does, the output is required to be a dictionary"""
        if len(self._out_mapping) > 1:
            return True
        else:
            return False

    def default(self, default: Any):
        """
        indicate a default value that we would like to have in case
        our function has an exception...

        (for example if we can not detect a language, because document is to short)
        """
        self._default = default
        return self

    def cache(self, allow_disk_cache=True):
        """indicate to document that we want this extractor function to be cached"""
        self._cache = True
        self._allow_disk_cache = allow_disk_cache
        return self

    def no_cache(self):
        self._cache = False
        return self

    @property
    def documentation(self):
        """return docs mapped to output of operator in pipeline"""
        if isinstance(self.__node_doc__, dict):
            return {v: self.__node_doc__[k] for k, v in self._out_mapping.items()}
        else:
            return self.__node_doc__ or ""

    def docs(self, doc_str: str = "", **kwdocstr: str):
        """
        Document the operator output.

        Be careful that the order of "docs" is the same as the output order of the operator.
        """
        # if our operator has multiple outputs, we can
        # document each individual output here...
        if kwdocstr:
            self.__node_doc__ = kwdocstr
        else:
            self.__node_doc__ = doc_str
        return self


class Alias(Operator):
    """
    Connect extractor variables with Aliases

    Alias will map "values" to "keys". So speaking in pipeline terms, "keys" are the downstream
    variable names.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.input(**{v: v for k, v in kwargs.items()})
        self.out(**{v: k for k, v in kwargs.items()})
        self.__node_doc__ = "Alias for: \n\n" + "\n".join(f"* {v}->{k} (output)" for k, v in kwargs.items())

    def __call__(self, **kwargs):
        return kwargs


class Constant(Operator):
    """declare one ore more constant values"""

    def __init__(self, **kwargs):
        super().__init__()
        self.const_result = kwargs
        self.out(**{k: k for k in kwargs})
        self.no_cache()
        self.__node_doc__ = "A constant value"

    def __call__(self):
        return self.const_result


CallableType = typing.TypeVar('CallableType')


class FunctionOperator(Operator[CallableType]):
    """Generic class which wraps an arbitrary
    function as an Operator

    It also automatically replaces the result of the function with
    a dictionary, if the function doesn't have multiple outputs in the pipeline
    otherwise the supplied function is required to return a dictionary with
    keys corresponding to the outputs in the pipeline.
    """

    def __init__(
            self,
            func: typing.Callable[..., CallableType],
            fallback_return_value: Any = None
    ):
        super().__init__()
        self._func = func
        self._default_return_value = fallback_return_value
        self.__node_doc__ = "No documentation"

    @functools.cached_property
    def return_type(self):
        """this property returns the type that was specified for the function operator"""
        if return_type := super().return_type:
            return return_type

        if type_hints := typing.get_type_hints(self._func):
            output_types = type_hints.get('return', typing.Any)
            typing_dict = self.map_output_types((output_types,))
            return typing_dict

        return self.map_output_types((typing.Any,))

    def __call__(self, *args, **kwargs) -> CallableType:
        try:
            res = self._func(*args, **kwargs)
            # return res
            if self.multiple_outputs:
                if not isinstance(res, dict):
                    raise TypeError(f"Function {self._func} does not return a dictionary, but has multiple outputs")
            else:
                output_key = next(iter(self._out_mapping))
                # check if we have a dictionary, if not, we will "make it one"
                if isinstance(res, dict):
                    # only use the result if it contains the output key
                    if output_key in res:
                        return res

                res = {output_key: res}

            return res
        except Exception as err:
            if self._default_return_value:
                return self._default_return_value
            else:
                raise err


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

        Configuration(qam_model_id='deepset/minilm-uncased-squad2'),
        QamExtractor()
            .pipe(property_dict="to_dict", trf_model_id="qam_model_id").out("answers").cache(),

    In this case, when calling a document we can dynamically configure the
    pipeline with the "qam_model_id" parameter:

        doc = Document(
            fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"),
            configuration=dict(qam_model_id='deepset/minilm-uncased-squad2'))
    """

    def __init__(self, **configuration_map):
        super().__init__()
        self._configuration_map = configuration_map
        self.no_cache()  # we don't need this, as everything is already saved in the _configuration_map
        # use the configuration map directly as output mapping
        self.out(*list(configuration_map.keys()))
        confdocs = '\n'.join(f"* {k} = {v} (default)" for k, v in configuration_map.items())
        self.__node_doc__ = f"Configuration for values:\n\n{confdocs}"

    def __call__(self):
        return self._configuration_map


class OperatorException(Exception):
    pass


class OperatorOutputException(OperatorException):
    pass
