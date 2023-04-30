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

from . import operators
from .list_utils import deep_str_convert

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True, slots=True)
class Font:
    name: str
    size: float
    color: str


class ElementType(Enum):
    Graphic = 1
    Line = 2
    Image = 3


@dataclass(slots=True)
class DocumentElement:
    type: ElementType
    p_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    rawtext: str | None
    font_infos: set[Font] | None
    linenum: int | None
    linewidth: float | None
    boxnum: int | None
    lineobj: Any | None
    gobj: Any | None
    non_stroking_color: str | None
    stroking_color: str | None
    stroke: bool | None
    fill: bool | None
    evenodd: int | None


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


# TODO: rename into "Operator"


class ConfigurationError(Exception):
    pass


class MetaPipelineClassConfiguration(type):
    """
    configures derived document class logic on construction.

    Also checks Operators etc...  for consistency. It sort of works like
    a poor-mans compiler or preprocessor.

    It basically takes the definition of the lazy pipeline
    in the "_extractors" variable
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

        if hasattr(new_class, "_extractors"):
            if new_class._extractors:
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
                # here. We need to do this here and not use the _x_funcs from parent class as
                # we need to make sure that new functions that were added to e.g. "*" also get added
                # to "*.pdf" and other document logic if we use the already calculated _x_funcs this
                # would not be guaranteed.

                uncombined_extractors: dict[str, dict[str, operators.Operator]] = {}
                extractor_combinations: dict[str, list[str]] = {}  # record the extraction hierarchy
                uncombined_x_configs: dict[str, dict[str, list[str]]] = {}
                ex: operators.Operator | str
                # loop through class hierarchy in order to get the logic of parent classes as well,
                # including the newly defined class
                for cl in reversed(class_hierarchy):
                    # loop through doc types
                    for doc_type, ex_list in cl._extractors.items():
                        doc_type_x_funcs = {}  # save function mappings for single doc_type
                        extractor_combinations[doc_type] = []  # save combination list for single doctype
                        doc_type_x_config = {}  # save configuration mappings for single doc_type
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
                                    doc_type_x_funcs[doc_key] = ex

                                    # build a map of configuration values for each
                                    # parameter. This means when a parameter gets called we know automatically
                                    # how to configure the corresponding Operator
                                    if ex._dynamic_config:
                                        doc_type_x_config[doc_key] = list(ex._dynamic_config.keys())

                        uncombined_extractors[doc_type] = uncombined_extractors.get(doc_type, {})
                        uncombined_extractors[doc_type].update(doc_type_x_funcs)
                        uncombined_x_configs[doc_type] = uncombined_x_configs.get(doc_type, {})
                        uncombined_x_configs[doc_type].update(doc_type_x_config)

                logger.debug("combining... extraction logic")
                # we need to re-initialize the class logic so that they are not linked
                # to the logic of the parent classes.
                new_class._x_funcs = {}
                new_class._x_config = {}
                doc_type: str
                # add all extractors by combining the logic for the different document types
                for doc_type in uncombined_extractors:
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
                    new_class._x_funcs[doc_type] = {}
                    new_class._x_config[doc_type] = {}

                    # build class combination in correct order:
                    # the first one is the least important, because it gets
                    # overwritten by subsequent classes
                    # TODO: get rid of the "standard" fallback...  for the pipeline
                    doc_type_order = ["*"]  # always use "*" as a fallback
                    if doc_type != "*":
                        doc_type_order += list(
                            reversed(extractor_combinations[doc_type])) + [doc_type]

                    # now save the x-functions/configurations in the _x_funcs dict
                    # (which might already exist from the parent class) in the correct order.
                    # already existing functions from the parent class get overwritten by the
                    # ones defined in the child class in "uncombined_extractors"
                    for ordered_doc_type in doc_type_order:
                        # add newly defined extractors overriding extractors defined
                        # in lower hierarchy logic
                        new_class._x_funcs[doc_type].update(uncombined_extractors[ordered_doc_type])
                        new_class._x_config[doc_type].update(uncombined_x_configs[ordered_doc_type])

                        # TODO: how do we add x-functions to

                # TODO: remove "dangling" extractors which lack input mapping

        else:
            raise ConfigurationError(f"no extractors defined in class {new_class}")

        elapsed = round((time() - start_time) * 1000, 4)
        logger.info(f"setting up Document class {new_class} took: {elapsed}ms")

        return new_class


class Pipeline(metaclass=MetaPipelineClassConfiguration):
    """
    This class is the base for all document classes in pydoxtools and
    defines a common pipeline interface for all.

    This class also defines a basic extraction schema which derived
    classes can override

    in order to create a new pipeline, the
    _extractor variable shoud be overwritten with a pipeline definition

    this pipeline will get compiled in a function mappint defined in
    _x_funcs
    """

    # TODO: use pandera (https://github.com/unionai-oss/pandera)
    #       in order to validate dataframes exchanged between extractors & loaders
    #       https://pandera.readthedocs.io/en/stable/pydantic_integration.html

    # TODO: how do we change extraction configuration "on-the-fly" if we have
    #       for example a structured document vs unstructered (PDF: unstructure,
    #       Markdown: structured)
    #       in this case table extraction algorithms for example would have to
    #       behave differently. We would like to use
    #       a different extractor configuration in that case...
    #       in other words: each extractor needs to be "conditional"

    # stores the extraction graph, a collection of connected functions
    # which extract data from a document
    # TODO: rename into pipeline
    _extractors: dict[str, list[operators.Operator]] = {}

    # a dict which provides access for all extractor functions by their "out-key"
    # which was defined in _extractors
    # TODO: rename into operators
    _x_funcs: dict[str, dict[str, operators.Operator]] = {}

    def __init__(self):
        self._cache_hits = 0
        self._x_func_cache: dict[operators.Operator, dict[str, Any]] = {}

    def config(self, **settings: dict[str, Any]) -> "Pipeline":
        """
        Set configuration parameters for a pipeline.

        This method loops through all "operators.Configure" instances in the pipeline
        and assigns the provided configuration settings to them.

        Args:
            **settings: A dictionary of key-value pairs representing the configuration
                settings for the pipeline. Each key is a string representing the name
                of the configuration setting, and the value is the corresponding value
                to be set.

        Returns:
            self: A reference to the current pipeline instance, allowing for method chaining.

        Example:
            pipeline = Pipeline()
            pipeline.config(param1=value1, param2=value2)
        """
        # Get all configuration objects in the pipeline
        configuration: dict[str, operators.Configuration] = \
            {k: v for k, v in self.x_funcs.items() if isinstance(v, operators.Configuration)}

        # Assign the settings to the corresponding configuration objects
        for k, v in settings.items():
            configuration[k]._configuration_map[k] = v

        # Return the current pipeline instance for method chaining
        return self

    @property
    def configuration(self) -> dict[str, Any]:
        """
        Gets all configuration objects of the pipeline and merges them into a single dictionary.

        Returns:
            dict: A dictionary containing the merged configuration objects of the pipeline, with keys as
                  the configuration names and values as the configuration objects.
        """
        configuration: dict[str, operators.Configuration] = \
            {k: v for k, v in self.x_funcs.items() if isinstance(v, operators.Configuration)}
        configuration_map = {}
        for c in configuration:
            configuration_map.update(**(configuration[c]._configuration_map))
        return configuration_map

    @cached_property
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
    def x_funcs(self) -> dict[str, operators.Operator]:
        """
        get all operators/pipeline nodes and their property names
        for this specific file type/pipeline
        """
        return self._x_funcs[self.pipeline_chooser]

    def property_dict(self, *args, **kwargs):
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

    def yaml(self, *args, **kwargs):
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
        out = self.property_dict(*args, **kwargs)
        out = yaml.safe_dump(out)
        return out

    def json(self, *args, **kwargs):
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
        out = self.property_dict(*args, **kwargs)
        out = json.dumps(out)
        return out

    def non_interactive_x_funcs(self) -> dict[str, operators.Operator]:
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
        for pipeline_id, ops in cls._x_funcs.items():
            for op_k, op in ops.items():
                oi: dict[str, set] = output_infos.get(op_k, None) or dict(pipe_types=set(), output_types=set())
                oi["pipe_types"].add(pipeline_id)
                if return_type := get_type_hints(op.__class__.__call__).get("return", None):
                    oi["output_types"].add(return_type)
                output_infos[op_k] = oi

        node_docs = []
        for k, v in output_infos.items():
            single_node_doc = f"""### {k}

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

    def x(self, extract_name: str, *args, **kwargs) -> Any:
        """
        Calls an extractor from the defined pipeline and returns the result.

        Args:
            extract_name (str): The name of the extractor to be called.
            *args: Variable-length argument list to be passed to the extractor.
            **kwargs: Arbitrary keyword arguments to be passed to the extractor.

        Returns:
            Any: The result of the extractor after processing the document.

        Raises:
            operators.OperatorException: If an error occurs while executing the extractor.

        Notes:
            The extractor's parameters can be overridden using *args and **kwargs.
        """
        if not (extractor_func := self.x_funcs.get(extract_name, None)):
            return self.__dict__[extract_name]  # choose the class' own properties as a fallback

        try:
            # check if we executed this function at some point...
            if extractor_func._cache:
                key = functools._make_key((extractor_func,) + args, kwargs, typed=False)
                # we need to check for "is not None" as we also have pandas dataframes in this
                # which cannot be checked for by simply using "if"
                if (res := self._x_func_cache.get(key, None)) is not None:
                    self._cache_hits += 1
                else:
                    res = extractor_func._mapped_call(self, *args, **kwargs)
                    self._x_func_cache[key] = res
            else:
                res = extractor_func._mapped_call(self, *args, **kwargs)

        except operators.OperatorException as e:
            logger.error(f"Extraction error in '{extract_name}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Extraction error in '{extract_name}': {e}")
            raise operators.OperatorException(f"could not get {extract_name} for {self}")
            # raise e

        return res[extract_name]

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

    def x_all(self):
        """
        Retrieves the results of all extractors defined in the pipeline.

        Returns:
            dict: A dictionary containing the results of all extractors, with keys as the extractor
                  names and values as the corresponding results.
        """
        return {property: self.x(property) for property in self.x_funcs}

    def run_all_extractors(self):
        """
        Runs all extractors defined in the pipeline for testing or pre-caching purposes.

        This method iterates through the defined extractors and calls each one, ensuring that the
        extractor logic is functioning correctly and caching the results if required.
        """
        # print(pdfdoc.elements)
        for x in self.x_funcs:
            self.x(x)

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
        Returns a string representation of the instance.

        This method provides a string representation of the instance, including the module and class
        names, as well as the file object or string and the source.

        Returns:
            str: A string representation of the instance.
        """
        if isinstance(self._fobj, str | bytes):
            return f"{self.__module__}.{self.__class__.__name__}({self._fobj[-10:]},{self.source})>"
        else:
            return f"{self.__module__}.{self.__class__.__name__}({self._fobj},{self.source})>"

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

    def pipeline_graph(self, image_path: str | pathlib.Path = None, document_logic_id="current"):
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
        if document_logic_id == "current":
            logic = self.x_funcs
        else:
            logic = self._x_funcs[document_logic_id]

        for name, f in logic.items():
            f_class = f.__class__.__name__ + "\n".join(f._out_mapping.keys())
            shape = "none"
            if isinstance(f, operators.Configuration):
                shape = "invhouse"
            graph.add_node(f_class, shape=shape)
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
