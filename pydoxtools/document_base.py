from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import dataclasses
import functools
import json
import logging
import pathlib
import pickle
import sys
import typing
import uuid
from enum import Enum
from functools import cached_property
from time import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pydantic
import spacy.tokens
import yaml
from diskcache import Cache

from . import operators_base
from .list_utils import deep_str_convert
from .operators_base import Operator, Configuration, OperatorException, OperatorOutputException
from .settings import settings

logger = logging.getLogger(__name__)

if sys.version_info.minor < 10:
    slot_args = dict()
else:
    slot_args = dict(slots=True)


@dataclasses.dataclass(eq=True, frozen=True, **slot_args)
class Font:
    name: str
    size: float
    color: str


class ElementType(Enum):
    Graphic = 1
    Text = 2
    Image = 3
    Table = 4
    TextBox = 5
    List = 6
    Header = 7
    Figure = 8


def convert_strings_to_enum_values(strings: list[str | Enum], enum):
    already_enums = [string for string in strings if isinstance(string, Enum)]
    enum_values = [getattr(enum, string, None) for string in strings if isinstance(string, str)]
    return [value for value in enum_values if value is not None] + already_enums


@dataclasses.dataclass(**slot_args)
class DocumentElement:
    type: ElementType
    labels: list[str] = dataclasses.field(
        default_factory=list)  # can be used to classify elements in multiple ways e.g. "address"
    p_num: int | list[int] = 0
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    i0: int | None = None  # in text based documents, this hints to the start character of the element in the raw text file
    i1: int | None = None  # in text based documenta, this hints to the end character of the element in the raw text file
    rawtext: str | None = None  # e.g. a formated html string
    text: str | None = None  # only the content, without document-type specific formatting or similar...
    sections: list[str] | None = None
    char_orientations: list[float] | None = None
    mean_char_orientation: float | None = None
    font_infos: set[Font] | None = None
    linenum: int | None = None
    linewidth: float | None = None
    boxnum: int | None = None
    obj: Any | None = None
    non_stroking_color: str | None = None
    stroking_color: str | None = None
    stroke: bool | None = None
    fill: bool | None = None
    evenodd: int | None = None
    level: int = 0
    id: int = None

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def place_holder_text(self):
        return f"{self.type.name}_{self.id}"


class TokenCollection:
    def __init__(self, tokens: list[spacy.tokens.Token]):
        self._tokens = tokens

    @cached_property
    def vector(self):
        return np.mean([t.vector for t in self._tokens], 0)

    @cached_property
    def text(self):
        return self.__str__()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, item):
        return self._tokens[item]

    def __str__(self):
        return " ".join(t.text for t in self._tokens)

    def __repr__(self):
        return "|".join(t.text for t in self._tokens)


class ConfigurationError(Exception):
    pass


class MetaPipelineClassConfiguration(type):
    """
    configures derived document class logic on construction.

    Also checks Operators etc...  for consistency. It sort of works like
    a poor-mans compiler or preprocessor.

    It basically takes the definition of the lazy pipeline
    in the "_operators" variable
    and maps it to function calls.

    TODO: can we achieve this in an easier way then using metaclasses?
          probably difficult ...
    """

    # TODO: we can probably refactor this function to make it easir to understand
    #       decouple etc...

    # in theory, we could add additional arguments to this function which we could
    # pass in our documentbase class
    def __new__(cls, clsname, bases, attrs):
        start_time = time()
        # construct our class
        new_class: Pipeline.__class__ = super().__new__(
            cls, clsname, bases, attrs)

        if hasattr(new_class, "_operators"):
            if new_class._operators:
                # TODO: add checks to make sure we don't have any name-collisions
                # configure class
                logger.info(f"configure {new_class} class...")

                # get all parent classes except the two document base definitions which don't
                # have a logic defined. We do this in order to incorporate pipelines
                # from parent classes
                class_hierarchy = new_class.mro()[:-2]

                # first we map all operators of the pipeline logic (and configurations)
                # into a dictionary using their outputs as a key and we do this for every defined pipeline and
                # also all pipelines in the parent class.
                #
                # This is similar to how we can access them in the compiled pipeline, but we will combine them
                # with a pipeline hierarchy/fall-back pipelines at a later stage to create
                # the final extraction logic. By overwriting pipelines which are lower in the tree hierarchy.
                #
                # We also combine _operator definitions from parent class and child class
                # here. We need to do this here and not use the _pipelines from parent class as
                # we need to make sure that new functions that were added to e.g. "*" also get added
                # to "application/pdf" and other document logic. If we use the already calculated _pipelines this
                # would not be guaranteed.

                uncombined_operators: dict[str, dict[str, Operator]] = {}
                pipeline_tree: dict[str, list[str]] = {}  # record the extraction hierarchy
                op: Operator | str
                # loop through class hierarchy in order to get the logic of parent classes as well,
                # including the newly defined class
                for cl in reversed(class_hierarchy):
                    # loop through doc types
                    for pipeline_type, operator_list in cl._operators.items():
                        doc_type_pipeline = {}  # save function mappings for single doc_type
                        pipeline_tree[pipeline_type] = []  # save combination list for single doctype
                        for op in operator_list:
                            # strings indicate that we would like to
                            # add all the functions from that document type as well but with
                            # lower priority
                            if isinstance(op, str):
                                pipeline_tree[pipeline_type].append(op)
                            else:
                                # go through all outputs of an extractor and
                                # map them to extraction variables inside document
                                # TODO: we could explicitly add the variables as property functions
                                #       which refer to the "x"-function in document?
                                for ex_key, doc_key in op._out_mapping.items():
                                    # input<->output mapping is already done i the extractor itself
                                    # check out Operator.pipe and Operator.map member functions
                                    doc_type_pipeline[doc_key] = op

                        uncombined_operators[pipeline_type] = uncombined_operators.get(pipeline_type, {})
                        uncombined_operators[pipeline_type].update(doc_type_pipeline)

                logger.debug("combining... extraction logic")
                # we need to re-initialize the class logic so that they are not linked
                # to the logic of the parent classes.
                new_class._pipelines = {}
                pipeline_type: str
                # add all Operators by combining the logic for the different document types
                for pipeline_type in uncombined_operators:
                    # first take our other document type and then add the current document type
                    # itself on top of it because of its higher priority overwriting
                    # Operators of the lower priority operators
                    # TODO: how do we make sure that we adhere to the tree structure?
                    #       we need to make sure that we generate the "lowest" priority (= top of tree)
                    #       document types first, and then subsequently until we are at the bottom
                    #       of the tree.

                    # TODO: add classes recursively (we can not combine logic blocks using multiple
                    #       levels right now). We probably need to run this function multiple times
                    #       in order for this to work.
                    new_class._pipelines[pipeline_type] = {}

                    # build class combination in correct order:
                    # the first one is the least important, because it gets
                    # overwritten by subsequent classes
                    # TODO: get rid of the "standard" fallback...  for the pipeline
                    doc_type_order = ["*"]  # always use "*" as a fallback
                    if pipeline_type != "*":
                        doc_type_order += list(
                            reversed(pipeline_tree[pipeline_type])) + [pipeline_type]

                    # now save the x-functions/configurations in the _pipelines dict
                    # (which might already exist from the parent class) in the correct order.
                    # already existing functions from the parent class get overwritten by the
                    # ones defined in the child class in "uncombined_operators"
                    for ordered_doc_type in doc_type_order:
                        # add newly defined operators overriding operators defined
                        # in lower hierarchy logic
                        new_class._pipelines[pipeline_type].update(uncombined_operators[ordered_doc_type])

                # TODO: remove "dangling" operators which lack input mapping

                # TODO: check if everything is alright here. like if types fit etc,
                #       if we have any "free" inputs which shouldnt be the case etc......
                for pipeline_name in new_class._pipelines:
                    for output_name, op in new_class._pipelines[pipeline_name].items():
                        # here, we want to make sure, that all Aliases inherit the type hints
                        # of their parents
                        if isinstance(op, operators_base.Alias):
                            for upstream_name in op._out_mapping:
                                parent_op: operators_base.Operator
                                if parent_op := new_class._pipelines[pipeline_name].get(upstream_name):
                                    if not op._output_type:
                                        op._output_type[upstream_name] = parent_op.return_type[upstream_name]


        else:
            raise ConfigurationError(f"no operators defined in class {new_class}")

        elapsed = round((time() - start_time) * 1000, 4)
        logger.info(f"setting up Document class {new_class} took: {elapsed}ms")

        return new_class


class DocumentLocation(pydantic.BaseModel):
    """Hints to a location in a document which """
    area: tuple[float, float, float, float] = pydantic.Field(
        None, description="These four number describe the boundingbox of an area in a document (e.g. pdf document)")
    page: int = None
    line: int = None
    range: tuple[int, int] = pydantic.Field(
        None, description="The range of the document string of this location (e.g. (10,40) -> fobj[10:40]) ")
    source: typing.Any = pydantic.Field(
        None, description="Source can be a file or URL or any other source of origin")


class Pipeline(metaclass=MetaPipelineClassConfiguration):
    """
    Base class for all document classes in pydoxtools, defining a common pipeline
    interface and establishing a basic pipeline schema that derived classes can override.

    The MetaPipelineClassConfiguration acts as a compiler to resolve the pipeline hierarchy,
    allowing pipelines to inherit, mix, extend, or partially overwrite each other.
    Each key in the _pipelines dictionary represents a different pipeline version.

    The [pydoxtools.Document][] class leverages this functionality to build separate pipelines for
    different file types, as the information processing requirements differ significantly
    between file types.

    Attributes:
        _operators (dict[str, list[pydoxtools.operators_base.Operator]]): Stores the definition of the
            pipeline graph, a collection of connected operators/functions that process
            data from a document.
        _pipelines (dict[str, dict[str, pydoxtools.operators_base.Operator]]): Provides access to all
            operator functions by their "out-key" which was defined in _operators.

    Todo:
        * Use pandera (https://github.com/unionai-oss/pandera) to validate dataframes
          exchanged between operators & loaders
          (https://pandera.readthedocs.io/en/stable/pydantic_integration.html)
    """

    _operators: dict[str, list[Operator]] = {}
    _pipelines: dict[str, dict[str, Operator]] = {}

    def __init__(self, **configuration):
        """
        Initializes the Pipeline instance with cache-related attributes.

        **configuration: A dictionary of key-value pairs representing the configuration
                settings for the pipeline. Each key is a string representing the name
                of the configuration setting, and the value is the corresponding value
                to be set.
        """
        self._configuration = {}
        self._configuration.update(configuration)
        self._stats = dict(
            cache_hits=0,
            disk_cache_hits=0,
            # this is a set, because the same error can appear many time
            cache_errors=set()  # log operators that are not cachable for debugging purposes
        )
        self._source = "base_pipeline"
        self._disk_cache_enable = settings.PDX_ENABLE_DISK_CACHE
        self._disk_cache_ttl = None

    def _pipeline_key(self):
        """
        represents a unique key for a pipeline with some specific source data
        this key can be used to cache results for a pipeline!
        """
        raise NotImplementedError("_pipeline_key function needs to be defined!!")

    def set_disk_cache_settings(
            self,
            enable: bool,
            ttl: int = 3600 * 24 * 7  # default cache expires after 1 week
    ):
        """Sets disk cache settings
        """
        self._disk_cache_ttl = ttl
        self._disk_cache_enable = enable
        return self

    @cached_property
    def _disk_cache_enabled(self):
        return self._disk_cache_enable

    @cached_property
    def _cache(self):
        return {}

    @cached_property
    def _disk_cache(self) -> dict[Operator, dict[str, Any]] | Cache:
        cache = Cache(settings.PDX_CACHE_DIR_BASE / "pipelines")
        return cache

    @property
    def configuration(self):
        """Returns a dictionary of all configuration objects for the current pipeline.

        Returns:
            dict: A dictionary containing the names and values of all configuration objects
                  for the current pipeline.
        """
        return {k: self.x(k) for k in self.get_configuration_names(self.pipeline_chooser)}

    @classmethod
    @functools.lru_cache
    def get_configuration_names(cls, pipeline: str) -> list[str]:
        """Returns a list of names of all configuration objects for a given pipeline.

        This is a cached function which is important,

        Args:
            pipeline (str): The name of the pipeline to retrieve configuration objects from.

        Returns:
            list: A list of strings containing the names of all configuration objects for the
                  given pipeline.
        """
        configuration: dict[str, Configuration] = \
            {k: v for k, v in cls._pipelines[pipeline].items() if isinstance(v, Configuration)}
        return list(configuration.keys())

    @property
    def pipeline_chooser(self) -> str:
        """
        Must be implemented by derived classes
        to decide which pipeline they should use.
        """
        # TODO: maybe rename this into "head-pipeline" or something like that?
        # TODO: not sure how to do this the "correct" way with @abstractmethod
        #       as we can not derive from ABC due to our metaclass...
        raise NotImplementedError("derived pipelines need to override this function!")

    @cached_property
    def x_funcs(self) -> dict[str, Operator]:
        """
        get all operators/pipeline nodes and their property names
        for this specific file type/pipeline
        """
        return self._pipelines[self.pipeline_chooser]

    def to_dict(self, *args, **kwargs):
        """
        Returns a dictionary that accumulates the properties given in *args or with a mapping in **kwargs.

        Args:
            *args (str): A variable number of strings, each representing a property name.
            **kwargs (dict): A dictionary mapping property names (values) to custom keys (keys) for the
                             returned dictionary.

        Note:
            This function currently only supports properties that do not require any arguments, such as
            "full_text". Properties like "answers" that return a function requiring arguments cannot be
            used with this function.

        Returns:
            dict: A dictionary with the accumulated properties and their values, using either the
                  property names or custom keys as specified in the input arguments.
        """

        properties = {a: getattr(self, a) for a in args}
        properties.update({v: getattr(self, k) for k, v in kwargs.items()})

        return deep_str_convert(properties)

    @functools.lru_cache
    def to_dataframe(self, *args, **kwargs):
        dictlist = self.to_dict(*args, **kwargs)
        if len(dictlist) == 1:
            name = next(iter(dictlist))
            df = pd.DataFrame(dictlist[name])
            df.name = name
        else:
            df = pd.DataFrame(dictlist)
        return df

    def df(self, *args, **kwargs):
        return self.to_dataframe(*args, **kwargs)

    def to_yaml(self, *args, **kwargs):
        """
        Returns a dictionary that accumulates the properties given in *args or with a mapping in **kwargs, and dumps the output as YAML.

        Args:
            *args (str): A variable number of strings, each representing a property name.
            **kwargs (dict): A dictionary mapping property names (values) to custom keys (keys) for the
                             returned dictionary.

        Note:
            This function currently only supports properties that do not require any arguments, such as
            "full_text". Properties like "answers" that return a function requiring arguments cannot be
            used with this function.

        Returns:
            str: A YAML-formatted string representing the accumulated properties and their values, using
                 either the property names or custom keys as specified in the input arguments.
        """
        out = self.to_dict(*args, **kwargs)
        out = yaml.safe_dump(out)
        return out

    def to_json(self, *args, **kwargs):
        """
        Returns a dictionary that accumulates the properties given in *args or with a
        mapping in **kwargs, and dumps the output as JSON.

        Args:
            *args (str): A variable number of strings, each representing a property name.
            **kwargs (dict): A dictionary mapping property names (values) to custom keys (keys) for the
                             returned dictionary.

        Note:
            This function currently only supports properties that do not require any arguments, such as
            "full_text". Properties like "answers" that return a function requiring arguments cannot be
            used with this function.

        Returns:
            str: A JSON-formatted string representing the accumulated properties and their values, using
                 either the property names or custom keys as specified in the input arguments.
        """
        out = self.to_dict(*args, **kwargs)
        out = json.dumps(out)
        return out

    def non_interactive_pipeline(self) -> dict[str, Operator]:
        """return all non-interactive operators/pipeline nodes"""
        NotImplementedError("TODO: search for functions that are type hinted as callable")

    @classmethod
    def node_infos(cls, pipeline_type=None) -> list[dict]:
        """
        Aggregates the pipeline operations and their corresponding types and metadata.

        This method iterates through all the pipelines registered in the class, and gathers
        information about each node/operation, such as the pipeline types it appears in,
        the return type of the operation, and the operation's docstring.

        Returns:
            TODO...
        """

        def get_dict_or_val(key: str, val: dict | Any):
            if isinstance(val, dict):
                return val.get(key, val)
            else:
                return val

        operator_infos: list[dict] = []
        pipelines = [pipeline_type] if pipeline_type else cls._pipelines.keys()
        for pipeline_id in pipelines:
            operators = cls._pipelines[pipeline_id]
            for operator_name, operator in operators.items():
                op_info = dict(
                    name=operator_name,
                    pipe_type=pipeline_id,
                    output_type=get_dict_or_val(operator_name, operator.return_type),
                    description=get_dict_or_val(operator_name, operator.documentation).strip(),
                    callable_params=getattr(operator, "callable_params", None),
                    operator_class=operator.__class__,
                    default_value=None
                )
                if isinstance(operator, operators_base.Configuration):
                    op_info["default_value"] = operator._configuration_map[operator_name]

                operator_infos.append(op_info)

        return operator_infos

    @classmethod
    def node_infos_agg(cls, as_string=True):
        output_infos = pd.DataFrame(cls.node_infos())

        # aggregate pipeline nodes info for documentation purposes
        # info: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#named-aggregation
        def aggregate_infos(x: pd.DataFrame):
            doc_df = x.groupby('description', sort=True).pipe_type.agg(lambda x: ", ".join(sorted(x))).reset_index()
            doc_table = doc_df[['pipe_type', 'description']].to_markdown(
                index=False)  # , tablefmt="grid") #tablefmt oesnt work for mkdocs
            doc_str = "\n\n---\n\n".join(doc_df.description)
            default_values = [""]
            node_type = ""
            if operators_base.Configuration in x.operator_class.to_list():
                default_values = sorted(i for i in set(x.default_value.apply(str)))
                node_type = "config"
            if as_string:
                default_values = " | ".join(default_values)
                return pd.Series({
                    "return_types": " | ".join(sorted(i for i in set(x.output_type.apply(str)))),
                    "pipeline_flows": ", ".join(sorted(x.pipe_type)),
                    "description": doc_table if (len(doc_df) > 1) else doc_str,
                    "description_str": doc_str,
                    "default_values": default_values,
                    "node_type": node_type
                })
            else:
                return pd.Series({
                    "return_types": set(x.output_type),
                    "pipeline_flows": sorted(x.pipe_type),
                    "description": doc_table if (len(doc_df) > 1) else doc_str,
                    "description_str": doc_str,
                    "default_values": default_values,
                    "node_type": node_type
                })

        node_docs = output_infos.groupby('name', sort=True).apply(aggregate_infos)
        return node_docs

    @classmethod
    def markdown_docs(cls):
        node_docs = cls.node_infos_agg()

        # sanitize output to work in html...
        node_docs = node_docs.applymap(lambda x: x.replace(">", r"\>").replace('*', r'\*'))

        # add functional operator nodes
        func_operator_docs = []
        for node_name in node_docs.loc[node_docs.node_type != 'config'].T:
            docs = node_docs.loc[node_name]
            func_operator_docs.append(f"""### {node_name}

{docs.description}

*name*
: `<{cls.__name__}>.x('{node_name}') or <{cls.__name__}>.{node_name}`

*return type*
: {docs.return_types}

*supports pipeline flows*
: {docs.pipeline_flows}""")

        # add configuration nodes
        configuration_docs = (node_docs.loc[node_docs.node_type == 'config']
                              .reset_index()[['name', 'description_str', 'default_values']]
                              .rename(columns={'description_str': 'description'})
                              .to_markdown(index=False))  # , tablefmt="grid"))
        docs = '\n\n'.join(func_operator_docs)
        return docs, configuration_docs

    def gather_inputs(self, mapped_args: dict[str, str], traceable: bool):
        """
        Gathers arguments from the pipeline and class, and maps them to the provided keys of kwargs.

        This method retrieves all required input parameters from _in_mapping, which was declared with "pipe".
        It first checks if the parameter is available as an extractor. If so, it calls the function to get the value.
        Otherwise, it gets the member-variables or functions of the derived pipeline
        class if an extractor with that name cannot be found.

        Args:
            **kwargs (dict): A dictionary containing the keys to be mapped to the corresponding values.

        Returns:
            dict: A dictionary containing the mapped keys and their corresponding values.
        """
        mapped_kwargs = {}
        # get all required input parameters from _in_mapping which was declared with "pipe"
        for k, v in mapped_args.items():
            # first check if parameter is available as an extractor
            mapped_kwargs[k] = self.x(v, traceable=traceable)
        return mapped_kwargs

    def x(self, operator_name: str, disk_cache: bool = False, traceable: bool = False) -> Any:
        """
        Calls an extractor from the defined pipeline and returns the result.

        Args:
            operator_name (str): The name of the extractor to be called.
            cache: if we want to cache the call. We can explicitly tell
                    the pipeline to cache a call. to make caching more efficient
                    by only caching the calls we want.
            traceable: Some operators will propagate the source of their information
                       through the pipeline. This adds traceability. By setting this to
                       traceable=True we can turn this feature on.

        Returns:
            Any: The result of the extractor after processing the document.

        Raises:
            operators.OperatorException: If an error occurs while executing the extractor.

        Notes:
            The extractor's parameters can be overridden using *args and **kwargs.
        """
        # override the operator if this instance has its own configuration
        if operator_name in self._configuration:
            return self._configuration[operator_name]
        elif not (operator_function := self.x_funcs.get(operator_name, None)):
            return super().__getattribute__(operator_name)  # choose the class' own properties as a fallback

        try:
            # whether function should be cached or not...
            finished_calculation = False
            use_disk_cache = self._disk_cache_enabled or disk_cache
            # taking the operator_function instead of the output as a key makes everything here more
            # efficient, because we don't have to store the output for individual
            # keys in case a function has multiple keys as an output...
            dict_cache_key = operator_function
            disk_cache_key = None
            if operator_function._cache:
                # first check if we already have the result in memory-cache,
                # as this is much faster than getting the result from disk
                # we can be less specific about this key as our cache here is
                # saved in the document instance.
                # As the arguments are always going to be the same,
                # only the operator name is sufficient here
                # we need to check for "is not None" as we also have pandas dataframes in this
                # which cannot be checked for by simply using "if"
                if (op_res := self._cache.get(dict_cache_key, None)) is not None:
                    self._stats["cache_hits"] += 1
                    finished_calculation = True

                # TODO: hash pandas.util.hash_pandas_object for mapped kwargs key
                if (not finished_calculation) and use_disk_cache \
                        and operator_function._allow_disk_cache:
                    # We are creating a key using a hash value
                    # for this specific instance of a pipeline object.
                    # this key should be provided by the pipeline itself.
                    # we can not use the function arguments as keys, as they are iteratively
                    # calculated through the pipeline. this means
                    # that we would have to calculate the entire tree in the pipeline to get the
                    # parameters for this function.
                    if disk_cache_key := self._pipeline_key():  # key might not exist!
                        disk_cache_key = operator_name + str(disk_cache_key)

                    if (not finished_calculation) and disk_cache_key:
                        try:
                            # we need to check for "is not None" as we also have pandas dataframes in this
                            # which cannot be checked for by simply using "if"
                            if (op_res := self._disk_cache.get(disk_cache_key, None)) is not None:
                                self._stats["disk_cache_hits"] += 1
                                finished_calculation = True
                        except (pickle.PicklingError) as error:
                            # simply don't do anything with the cache, if we can not cache it, but log
                            # the function;)
                            self._stats["cache_errors"].add((operator_name, error))
                    else:
                        self._stats["cache_errors"].add((operator_name, "no key for disk caching"))

            # if we haven't gotten the result from cache yet, calculate it! ...
            if finished_calculation == False:
                mapped_kwargs = self.gather_inputs(operator_function._in_mapping, traceable=traceable)
                mapped_kwargs = {k: (v.data if isinstance(v, operators_base.OperatorResult) else v)
                                 for k, v in mapped_kwargs.items()}
                if operator_function._default:
                    try:
                        op_res = operator_function(**mapped_kwargs)
                    except:
                        op_res = operator_function._default
                else:
                    op_res = operator_function(**mapped_kwargs)
                # and save the result in both caches
                if operator_function._cache:
                    self._cache[dict_cache_key] = op_res
                    if use_disk_cache and disk_cache_key:
                        try:
                            # self._disk_cache.set(disk_cache_key, op_res, expire=self._disk_cache_ttl)
                            self._disk_cache.set(disk_cache_key, op_res)
                        except (pickle.PicklingError, AttributeError, TypeError) as error:
                            self._stats["cache_errors"].add((operator_name, error))

            # extract data from OperatorResult
            if isinstance(op_res, operators_base.OperatorResult):
                res = op_res.data
                source = op_res.source
            else:
                res = op_res
                source = self

            # TODO: this can probably made more elegant
            # TODO: get rid of "dict" results...
            if operator_function.multiple_outputs:
                res = {operator_function._out_mapping[k]: v for k, v in res.items()
                       if k in operator_function._out_mapping}
            else:
                # use first key of out_mapping for output if
                # we only have a single return value
                res = {next(iter(operator_function._out_mapping.values())): res}

        except OperatorException as e:
            logger.error(f"Extraction error in {self}, '{operator_name}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Extraction error in {self}, '{operator_name}': {e}")
            raise OperatorException(f"{type(e)}: could not get '{operator_name}' for {self}")
            # raise e

        # TODO automatically wrap the result with functools.cache if
        #      it is a callable. (or use our own cache decorator)

        try:
            final_result = res[operator_name]
        except KeyError:
            raise OperatorOutputException(
                f"Key '{operator_name}' does not exist in output dict of the Operator {operator_function}"
                f"it might be necessary to return the result of the operator "
                f"as a dictionary with the key '{operator_name}' if the operator has multiple outputs."
            )

        if traceable:
            return operators_base.OperatorResult(
                data=final_result,
                source=source
            )
        else:
            return final_result

    def get(self, property: str, default_return: Any = None) -> Any:
        try:
            return self.x(property)
        except KeyError:
            return default_return

    @classmethod
    def operator_types(cls):
        """
        This function returns a dictionary of operators with their types
        which is suitable for declaring a pydantic model.

        only_valid_json_schema_types:    if this is set to True, we make sure that only valid json
                                         schema types are included in the model.
                                         The typical use case is to expose the pipeline via this
                                         model to an http API e.g. through fastapi. In this
                                         case we should only allow types that are valid json schema.
                                         Therefore, this is set to "False" by default.
        """

        def test_json_schema(x):
            try:
                pydantic.schema_json_of(x)
                return True
            except:
                # if a model should be a valid json schema, omit the definition
                return False

        # get types
        types = cls.node_infos_agg(as_string=False).copy()
        types['return_types_union'] = types.return_types.apply(lambda x: typing.Union[tuple(x)])
        types['works_as_json_schema'] = types.return_types_union.apply(test_json_schema)
        return types

    @classmethod
    def Model(cls):
        operator_signatures = cls.operator_types()

        class Config:
            arbitrary_types_allowed = True

        PydanticModel = pydantic.create_model(cls.__name__, **operator_signatures, __config__=Config)
        return PydanticModel

    def __getitem__(self, extract_name) -> Any:
        """
        Retrieves an extractor result by directly accessing it as an attribute.

        This method is automatically called for attribute names that
        aren't defined on class level, allowing for a convenient
        way to access pipeline operator outputs without needing to call the 'x' method.

        Example:
            >>> document["addresses"]
            instead of document.x('addresses')

        Args:
            extract_name (str): The name of the extractor result to be accessed.

        Returns:
            Any: The result of the extractor after processing the document.
        """
        return self.x(extract_name)

    def __getattr__(self, extract_name) -> Any:
        """
        Retrieves an extractor result by directly accessing it as an attribute.

        This method is automatically called for attribute names that
        aren't defined on class level, allowing for a convenient
        way to access pipeline operator outputs without needing to call the 'x' method.

        Example:
            >>> document.addresses
            instead of document.x('addresses')

        Args:
            extract_name (str): The name of the extractor result to be accessed.

        Returns:
            Any: The result of the extractor after processing the document.
        """
        return self.x(extract_name)

    def __getstate__(self):
        """
        return necessary variables for pickling, ensuring that
        we leave out everything that can potentially have
        a lambda function in it...
        """
        state = self.__dict__.copy()
        drop_vars = [
            "x_funcs", "_pipelines", "_cache",
            "_disk_cache", "_operators",
        ]
        for v in drop_vars:
            state.pop(v, None)
        return state

    def __setstate__(self, state: dict):
        """
        we need to restore _x_func_cache for pickling to work...
        """
        # for k,v in state:
        self.__dict__.update(state)
        # TODO: restore more cached values to increase speed in a distributed setting.
        #       for this we need to rely on our cache to work with strings as keys
        #       and not functions...

    def x_all(self):
        """
        Retrieves the results of all operators defined in the pipeline.

        Returns:
            dict: A dictionary containing the results of all operators, with keys as the extractor
                  names and values as the corresponding results.
        """
        return {property: self.x(property) for property in self.x_funcs}

    def run_pipeline(self, exclude: list[str] = None):
        """
        Runs all operators defined in the pipeline for testing or pre-caching purposes.

        !!IMPORTANT!!!  This function should normally not be used as the pipeline is lazily executed
        anyway.

        This method iterates through the defined operators and calls each one, ensuring that the
        extractor logic is functioning correctly and caching the results if required.
        """
        # print(pdfdoc.elements)
        exclude = exclude or []
        for x in self.x_funcs:
            if x not in exclude:
                self.x(x)

    def run_pipeline_fast(self):
        """run pipeline, but exclude long-running calculations"""
        self.run_pipeline(exclude=["slow_summary"])

    def pre_cache(self):
        """
        Pre-caches the results of all operators that have caching enabled.

        This method iterates through the defined operators and calls each one with caching enabled,
        storing the results for faster access in future calls.

        Returns:
            self: The instance of the class, allowing for method chaining.
        """

        for x, ex in self.x_funcs:
            if ex._cache:
                self.x(x)

        return self

        # TODO: save document structure as a graph...
        # nx.write_graphml_lxml(G,'test.graphml')
        # nx.write_graphml(G,'test.graphml')

    def __repr__(self):
        """
        Returns:
            str: A string representation of the instance.
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    # TODO: more configuration options:
    #       - which nlp models (spacy/transformers) to use
    #       - should "full text" include tables?
    #       - should ner include tables/figures?

    # TODO: calculate md5-hash for the document and
    #       use __eq__ with that hash...
    #       we need this for caching purposes but also in order
    #       check if a document already exists...

    @cached_property
    def uuid(self):
        """
        Retrieves a universally unique identifier (UUID) for the instance.

        This method generates a new UUID for the instance using Python's `uuid.uuid4()` function. The
        UUID is then cached as a property, ensuring that the same UUID is returned for subsequent
        accesses.

        Returns:
            uuid.UUID: A unique identifier for the instance.
        """
        return uuid.uuid4()

    @classmethod
    def pipeline_graph(cls, image_path: str | pathlib.Path = None, document_logic_id="*"):
        """
        Generates a visualization of the defined pipelines and optionally saves it as an image.

        Args:
            image_path (str | pathlib.Path, optional): File path for the generated image. If provided, the
                                                       generated graph will be saved as an image.
            document_logic_id (str, optional): The document logic ID for which the pipeline graph should
                                               be generated. Defaults to "current".

        Returns:
            AGraph: A PyGraphviz AGraph object representing the pipeline graph. This object can be
                    visualized or manipulated using PyGraphviz functions.

        Notes:
            This method requires the NetworkX and PyGraphviz libraries to be installed.
        """
        # TODO: change into a static method
        graph = nx.DiGraph()
        logic = cls._pipelines[document_logic_id]

        for name, operator in logic.items():
            f_class = "\n".join([operator.__class__.__name__] + list(operator._out_mapping.keys()))
            name = operator.name()
            shape = "none"
            if isinstance(operator, Configuration):
                shape = "invhouse"
            graph.add_node(f_class, shape=shape, label=name)
            # out-edges
            for k, v in operator._out_mapping.items():
                graph.add_node(v)  # , shape="none")
                graph.add_edge(f_class, v)
            for k, v in operator._in_mapping.items():
                graph.add_edge(v, f_class)
            # f._dynamic_config

        dotgraph = nx.nx_agraph.to_agraph(graph)
        if image_path:
            dotgraph.layout('dot')  # Use the 'dot' layout engine from Graphviz
            # Save the graph as an image
            dotgraph.draw(str(image_path.absolute()))

        return dotgraph
