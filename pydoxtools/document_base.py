import functools
import json
import logging
import pathlib
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from time import time
from typing import List, Any, get_type_hints

import networkx as nx
import numpy as np
import spacy.tokens
import yaml
from diskcache import Cache

from .list_utils import deep_str_convert
from .operators_base import Operator, Configuration, OperatorException, OperatorOutputException
from .settings import settings

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True, slots=True)
class Font:
    name: str
    size: float
    color: str


class ElementType(Enum):
    Graphic = 1
    Text = 2
    Image = 3


@dataclass(slots=True)
class DocumentElement:
    type: ElementType
    p_num: int | None = None
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    rawtext: str | None = None
    text: str | None = None
    sections: list[str] | None = None
    font_infos: set[Font] | None = None
    linenum: int | None = None
    linewidth: float | None = None
    boxnum: int | None = None
    lineobj: Any | None = None
    gobj: Any | None = None
    non_stroking_color: str | None = None
    stroking_color: str | None = None
    stroke: bool | None = None
    fill: bool | None = None
    evenodd: int | None = None
    level: int = 0


class TokenCollection:
    def __init__(self, tokens: List[spacy.tokens.Token]):
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
        new_class: Pipeline.__class__ = super(MetaPipelineClassConfiguration, cls).__new__(
            cls, clsname, bases, attrs)

        if hasattr(new_class, "_operators"):
            if new_class._operators:
                # TODO: add checks to make sure we don't have any name-collisions
                # configure class
                logger.info(f"configure {new_class} class...")

                # get all parent classes except the two document base definitions which don't
                # have a logic defined
                class_hierarchy = new_class.mro()[:-2]

                # first we map all functions of the extraction logic (and configurations)
                # into a dictionary using their outputs as a key and we do this for every defined filetype and
                # parent class.
                # This is similar to how we can access them in the final class, but we will combine them
                # with fall-back-filetypes at a later stage to create the final extraction logic
                #
                # We also combine _extractor definitions from parent class and child class
                # here. We need to do this here and not use the _pipelines from parent class as
                # we need to make sure that new functions that were added to e.g. "*" also get added
                # to "*.pdf" and other document logic if we use the already calculated _pipelines this
                # would not be guaranteed.

                uncombined_operators: dict[str, dict[str, Operator]] = {}
                extractor_combinations: dict[str, list[str]] = {}  # record the extraction hierarchy
                ex: Operator | str
                # loop through class hierarchy in order to get the logic of parent classes as well,
                # including the newly defined class
                for cl in reversed(class_hierarchy):
                    # loop through doc types
                    for doc_type, ex_list in cl._operators.items():
                        doc_type_pipeline = {}  # save function mappings for single doc_type
                        extractor_combinations[doc_type] = []  # save combination list for single doctype
                        for ex in ex_list:
                            # strings indicate that we would like to
                            # add all the functions from that document type as well but with
                            # lower priority
                            if isinstance(ex, str):
                                extractor_combinations[doc_type].append(ex)
                            else:
                                # go through all outputs of an extractor and
                                # map them to extraction variables inside document
                                # TODO: we could explicitly add the variables as property functions
                                #       which refer to the "x"-function in document?
                                for ex_key, doc_key in ex._out_mapping.items():
                                    # input<->output mapping is already done i the extractor itself
                                    # check out Operator.pipe and Operator.map member functions
                                    doc_type_pipeline[doc_key] = ex

                        uncombined_operators[doc_type] = uncombined_operators.get(doc_type, {})
                        uncombined_operators[doc_type].update(doc_type_pipeline)

                logger.debug("combining... extraction logic")
                # we need to re-initialize the class logic so that they are not linked
                # to the logic of the parent classes.
                new_class._pipelines = {}
                doc_type: str
                # add all extractors by combining the logic for the different document types
                for doc_type in uncombined_operators:
                    # first take our other document type and then add the current document type
                    # itself on top of it because of its higher priority overwriting
                    # extractors of the lower priority extractors
                    # TODO: how do we make sure that we adhere to the tree structure?
                    #       we need to make sure that we generate the "lowest" priority (= top of tree)
                    #       document types first, and then subsequently until we are at the bottom
                    #       of the tree.

                    # TODO: add classes recursively (we can not combine logic blocks using multiple
                    #       levels right now). We probably need to run this function multiple times
                    #       in order for this to work.
                    new_class._pipelines[doc_type] = {}

                    # build class combination in correct order:
                    # the first one is the least important, because it gets
                    # overwritten by subsequent classes
                    # TODO: get rid of the "standard" fallback...  for the pipeline
                    doc_type_order = ["*"]  # always use "*" as a fallback
                    if doc_type != "*":
                        doc_type_order += list(
                            reversed(extractor_combinations[doc_type])) + [doc_type]

                    # now save the x-functions/configurations in the _pipelines dict
                    # (which might already exist from the parent class) in the correct order.
                    # already existing functions from the parent class get overwritten by the
                    # ones defined in the child class in "uncombined_operators"
                    for ordered_doc_type in doc_type_order:
                        # add newly defined extractors overriding extractors defined
                        # in lower hierarchy logic
                        new_class._pipelines[doc_type].update(uncombined_operators[ordered_doc_type])

                # TODO: remove "dangling" extractors which lack input mapping

        else:
            raise ConfigurationError(f"no extractors defined in class {new_class}")

        elapsed = round((time() - start_time) * 1000, 4)
        logger.info(f"setting up Document class {new_class} took: {elapsed}ms")

        return new_class


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
          exchanged between extractors & loaders
          (https://pandera.readthedocs.io/en/stable/pydantic_integration.html)
    """

    _operators: dict[str, list[Operator]] = {}
    _pipelines: dict[str, dict[str, Operator]] = {}

    def __init__(self):
        """
        Initializes the Pipeline instance with cache-related attributes.

        Attributes:
            _cache_hits (int): Number of cache hits during pipeline execution.
            _x_func_cache (dict[pydoxtools.operators_base.Operator, dict[str, Any]]): Cache for operator
                functions to store intermediate results.
        """
        self._configuration = {}
        self._cache_hits = 0
        self._source = "base_pipeline"
        self._x_func_cache: dict[Operator, dict[str, Any]] = {}
        self._cache_type = "auto"

    def set_cache_type(self, cache_type: str):
        """Sets cache type to be on disk or in memory. Unless
        it is set to "auto", it will override th global settings.
        
        Values can be:  
        - "auto" this uses the cache as specified in the global settings
        - "memory" (default)
        - "diskcache"  which uses diskcache as a caching backend
        """

        self._cache_type = cache_type
        return self

    @cached_property
    def cache(self):
        if self._cache_type == "auto":
            cache_mode = settings.PDX_CACHE_TYPE
        else:
            cache_mode = self._cache_type

        if cache_mode == "memory":
            return self._x_func_cache
        elif cache_mode == "diskcache":
            cache = Cache(settings.PDX_CACHE_DIR_BASE / "pipelines")
            return cache
        else:
            raise NotImplementedError(f"Cache mode: {cache_mode} not available!")

    def config(self, **configuration: Any) -> "Pipeline":
        """
        Set configuration parameters for a pipeline.

        Args:
            **configuration: A dictionary of key-value pairs representing the configuration
                settings for the pipeline. Each key is a string representing the name
                of the configuration setting, and the value is the corresponding value
                to be set.

        Returns:
            self: A reference to the current pipeline instance, allowing for method chaining.

        Example:
            pipeline = Pipeline()
            pipeline.config(param1=value1, param2=value2)
        """
        # replace configuration objects in the pipeine with
        # our own configuration objects with correct keys
        self._configuration = configuration

        # Return the current pipeline instance for method chaining
        return self

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
        """return all non-interactive extractors/pipeline nodes"""
        NotImplementedError("TODO: search for functions that are type hinted as callable")

    @classmethod
    def pipeline_docs(cls):
        """
        Returns a formatted string containing the documentation for each pipeline operation in the class.

        This class method iterates through the pipeline operations, collects information about their
        output types and supported pipelines, and formats the documentation accordingly.

        Returns:
            str: A formatted string containing the documentation for each pipeline operation, including
                 operation name, usage, return type, and supported pipelines.
        """
        output_infos = {}
        # aggregate information
        for pipeline_id, ops in cls._pipelines.items():
            for op_k, op in ops.items():
                oi: dict[str, set] = output_infos.get(op_k, None) or dict(pipe_types=set(), output_types=set())
                oi["pipe_types"].add(pipeline_id)
                if return_type := get_type_hints(op.__class__.__call__).get("return", None):
                    oi["output_types"].add(return_type)
                oi["description"] = op.__node_doc__
                output_infos[op_k] = oi

        node_docs = []
        for k, v in output_infos.items():
            single_node_doc = f"""### {k}
            
{v['description']}

Can be called using:

    doc.x('{k}')
    # or
    doc.{k}

return type
: {"".join(sorted(str(i) for i in v['output_types']))}

supports pipelines
: {",".join(sorted(v['pipe_types']))}"""
            node_docs.append(single_node_doc)

        docs = '\n\n'.join(node_docs)
        return docs

    def gather_arguments(self, **kwargs):
        """
        Gathers arguments from the pipeline and class, and maps them to the provided keys of kwargs.

        This method retrieves all required input parameters from _in_mapping, which was declared with "pipe".
        It first checks if the parameter is available as an extractor. If so, it calls the function to get the value.
        Otherwise, it gets the "native" member-variables or other functions if an extractor with that name is not found.

        Args:
            **kwargs (dict): A dictionary containing the keys to be mapped to the corresponding values.

        Returns:
            dict: A dictionary containing the mapped keys and their corresponding values.
        """
        mapped_kwargs = {}
        # get all required input parameters from _in_mapping which was declared with "pipe"
        for k, v in kwargs.items():
            # first check if parameter is available as an extractor
            if v in self.x_funcs:
                # then call the function to get the value
                mapped_kwargs[k] = self.x(v)
            else:
                # get "native" member-variables or other functions
                # if not found an extractor with that name
                mapped_kwargs[k] = getattr(self, v)
        return mapped_kwargs

    def x(self, operator_name: str) -> Any:
        """
        Calls an extractor from the defined pipeline and returns the result.

        Args:
            operator_name (str): The name of the extractor to be called.

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
            return self.__dict__[operator_name]  # choose the class' own properties as a fallback

        try:
            # gather keywords for function from inputs
            mapped_kwargs = self.gather_arguments(**operator_function._in_mapping)

            # check if we executed this function at some point...
            if operator_function._cache:  # whether function should be cached or not...
                # TODO: implement our own key function in order
                #       to be able to pickle the document cache!
                #
                # {} for kwargs in _make_key is only temporary to pass tests with class-bound tests where
                # we don't need the additional key parameters
                key = functools._make_key((operator_function,), {}, typed=False)
                # key = (operator_function,)
                # TODO: add parameters here as well so that w can use a "central" diskcache...
                #       and different instances of pipelines don't mix up their cached
                #       functions...
                # key = self.__class__.__name__ + operator_name
                # we need to check for "is not None" as we also have pandas dataframes in this
                # which cannot be checked for by simply using "if"
                if (res := self.cache.get(key, None)) is not None:
                    self._cache_hits += 1
                else:
                    res = operator_function(**mapped_kwargs)
                    self.cache[key] = res
            else:
                res = operator_function(**mapped_kwargs)

            # TODO: this can probably made more elegant
            if isinstance(res, dict):
                res = {operator_function._out_mapping[k]: v for k, v in res.items()
                       if k in operator_function._out_mapping}
            else:
                # use first key of out_mapping for output if
                # we only have a single return value
                res = {next(iter(operator_function._out_mapping)): res}


        except OperatorException as e:
            logger.error(f"Extraction error in '{operator_name}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Extraction error in '{operator_name}': {e}")
            raise OperatorException(f"could not get {operator_name} for {self}")
            # raise e

        # TODO automatically wrap the result with functools.cache if
        #      it is a callable. (or use our own cache decorator)

        try:
            return res[operator_name]
        except KeyError:
            raise OperatorOutputException(
                f"Key '{operator_name}' does not exist in output dict of Operator {operator_function}")

    def get(self, property: str, default_return: Any = None) -> Any:
        try:
            return self.x(property)
        except KeyError:
            return default_return

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
        we leave out everything that can potentiall have some sort
        of a lambda function in it...
        """
        state = self.__dict__.copy()
        state.pop("_x_func_cache", None)
        state.pop("x_funcs", None)
        state.pop("_pipelines", None)
        return state

    def __setstate__(self, state: dict):
        """
        we need to restore _x_func_cache for pickling to work...
        """
        # for k,v in state:
        self.__dict__.update(state)
        self._x_func_cache = {}
        # TODO: restore more cached values to increase speed in a distributed setting.
        #       for this we need to rely on our cache to work with strings as keys
        #       and not functions...

    def x_all(self):
        """
        Retrieves the results of all extractors defined in the pipeline.

        Returns:
            dict: A dictionary containing the results of all extractors, with keys as the extractor
                  names and values as the corresponding results.
        """
        return {property: self.x(property) for property in self.x_funcs}

    def run_pipeline(self, exclude: list[str] = None):
        """
        Runs all extractors defined in the pipeline for testing or pre-caching purposes.

        !!IMPORTANT!!!  This function should normally not be used as the pipeline is lazily executed
        anyway.

        This method iterates through the defined extractors and calls each one, ensuring that the
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
        Pre-caches the results of all extractors that have caching enabled.

        This method iterates through the defined extractors and calls each one with caching enabled,
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

        for name, f in logic.items():
            f_class = "\n".join([f.__class__.__name__] + list(f._out_mapping.keys()))
            name = f.__class__.__name__
            shape = "none"
            if isinstance(f, Configuration):
                shape = "invhouse"
            graph.add_node(f_class, shape=shape, label=name)
            # out-edges
            for k, v in f._out_mapping.items():
                graph.add_node(v)  # , shape="none")
                graph.add_edge(f_class, v)
            for k, v in f._in_mapping.items():
                graph.add_edge(v, f_class)
            # f._dynamic_config

        dotgraph = nx.nx_agraph.to_agraph(graph)
        if image_path:
            dotgraph.layout('dot')  # Use the 'dot' layout engine from Graphviz
            # Save the graph as an image
            dotgraph.draw(str(image_path.absolute()))

        return dotgraph
