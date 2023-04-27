import abc
import functools
import logging
import pathlib
import typing
import uuid
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from time import time
from typing import List, Any

import networkx as nx
import numpy as np
import spacy.tokens

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


class ExtractorException(Exception):
    pass


# TODO: rename into "Operator"
class Extractor(ABC):
    """
    Base class to build extraction logic for information extraction from
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
    to set some input parameters to this function manually.
    If the same parameters are also set in doc.pipe the parameters are
    optional and will only be taken if explicitly set through doc.config(...).

        doc.dynamic()

    This function can be accessed through:

        doc.config(my_dynamic_parameter="some_new_value")

    """

    # TODO:  how can we anhance the type checking for outputs?
    #        maybe turn this into a dataclass?

    def __init__(self):
        # try to keep __init__ with no arguments for Extractor..
        self._in_mapping: dict[str, str] = {}
        self._out_mapping: dict[str, str] = {}
        self._cache = False  # TODO: switch to "True" by default
        self._dynamic_config: dict[str, str] = {}
        self._interactive = False

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, typing.Any] | Any:
        pass

    def _mapped_call(
            self, parent_document: "Pipeline",
            *args,
            config_params: dict[str, Any] = None,
            **kwargs
    ) -> dict[
        str, typing.Any]:
        """
        map objects from document properties to
        processing function.

        essentially, This maps the outputs from one element of _extractors
        to the inputs of another and also makes sure
        to override certain parameters that were specified in
        a config when calling the document class.

        argument precedence is as follows:

        python-class-member < extractor-graph-function < config

        # TODO: maybe we should change precedence and make config the lowest?
        """
        mapped_kwargs = {}
        # get all required input parameters from _in_mapping which was declared with "pipe"
        for k, v in self._in_mapping.items():
            # first check if parameter is available as an extractor
            if v in parent_document.x_funcs:
                # then call the function to get the value
                mapped_kwargs[k] = parent_document.x(v)
            else:
                # get "native" member-variables or other functions
                # if not found an extractor with that name
                mapped_kwargs[k] = getattr(parent_document, v)

        # get potential overriding parameters to override function call from _dynamic_config
        if config_params:
            override_parameters = {self._dynamic_config[k]: v for k, v in config_params.items()}
            mapped_kwargs.update(override_parameters)

        # override graph args directly with function call params...
        mapped_kwargs.update(kwargs)
        output = self(*args, **mapped_kwargs)
        if isinstance(output, dict):
            return {self._out_mapping[k]: v for k, v in output.items() if k in self._out_mapping}
        else:
            # use first key of out_mapping for output if
            # we only have a single return value
            return {next(iter(self._out_mapping)): output}

    def config(self, *args, **kwargs):
        # TODO: list mandatory configuration arguments!
        # TODO: check for mandatory configuration arguments during pipeline instantiation
        """
        Function is used to declare parameters of the extractor "configurable".
        This means on instantiation of a pipeline, this parameter can be changed.

        If the parameter of the extractor is:

            - not connected using the "pipe" argument
            - or does not have a "default" value set in the python function
              parameters

        it HAS to to be specified using the config argumetn during pipeline instantiation

        The Question & Answering part of the pydoxtools.Document class
        was specified with this config function like this:

            QamExtractor(model_id=settings.PDXT_STANDARD_QAM_MODEL)
                .pipe(text="full_text").out("answers").cache().config(trf_model_id="qam_model_id"),

        In this case, when calling a document we can dynamically configure the
        pipeline with the "qam_model_id" parameter:

            doc = Document(
                fobj=doc_str, document_type=".pdf"
            ).config(dict(qam_model_id='deepset/roberta-base-squad2'))

        This function gets called during pipeline definition in a document class
        in order to make some extractor arguments dynamic and changeable.

        The configured parameters for the pipeline will get sent to the extractor
        function on every call (xecept if its cached).

        Essentially it works similar to functools.partial.

        *args and **kwargs specify which function calls should be called from
        a config file.

        e.g. calling:

            EXTRACTOR.config(trf_model_id="qam_model_id")

        means, that the parameter "trf_model_id" can be specified by initializing the document
        class with a config parameter "qam_model_id" which then gets mapped to the
        function parameter "trf_model_id".

        "document(..., config=dict(trf_model_id='distilbert-base-cased-distilled-squad'))

        This function call can still be overwritten
        by calling the function through "x" directly using *args or **kwargs.

        The reason Extractors can be configured this way is, that the Extractors
        will already be initialized on Class definition in the MetaPipelineClassConfiguration
        for efficiency reasons. But we would also like to have an easy way to set
        certain paramaters on Instantiation of a pipeline. This can be done through this config
        function.
        """
        self._dynamic_config = {v: k for k, v in kwargs.items()}
        self._dynamic_config.update({k: k for k in args})

        return self

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

    def cache(self):
        """indicate to document that we want this extractor function to be cached"""
        self._cache = True
        return self

    def no_cache(self):
        self._cache = False
        return self


class ConfigurationError(Exception):
    pass


class MetaPipelineClassConfiguration(type):
    """
    configures derived document class logic on construction.

    Also checks Extractors etc...  for consistency. It sort of works like
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

                uncombined_extractors: dict[str, dict[str, Extractor]] = {}
                extractor_combinations: dict[str, list[str]] = {}  # record the extraction hierarchy
                uncombined_x_configs: dict[str, dict[str, list[str]]] = {}
                ex: Extractor | str
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
                                    # check out Extractor.pipe and Extractor.map member functions
                                    doc_type_x_funcs[doc_key] = ex

                                    # build a map of configuration values for each
                                    # parameter. This means when a parameter gets called we know automatically
                                    # how to configure the corresponding Extractor
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
    _extractors: dict[str, list[Extractor]] = {}

    # a dict which provides access for all extractor functions by their "out-key"
    # which was defined in _extractors
    _x_funcs: dict[str, dict[str, Extractor]] = {}

    # dict which stores function configurations
    _x_config: dict[str, dict[str, dict[str, Any]]] = {}

    def __init__(self):
        self._cache_hits = 0
        self._x_func_cache: dict[Extractor, dict[str, Any]] = {}
        self._config = {}

    def config(self, **kwargs):
        """Set a standard configuration for a pipeline"""
        self._config = kwargs
        return self

    @cached_property
    def pipeline_chooser(self) -> str:
        # TODO: not sure how to do this the "correct" way with @abstractmethod
        #       as we can not derive from ABC due to our metaclass...
        raise NotImplementedError("derived pipelines need to override this function!")

    @cached_property
    def x_funcs(self) -> dict[str, Extractor]:
        """
        get all extractors and their property names for this specific file type
        """
        return self._x_funcs[self.pipeline_chooser]

    def property_dict(self, *args, **kwargs):
        """
        return a dictionary which accumulates the properties given
        in *args or with a mapping in **kwargs where the keys in kwargs are the
        variable in the returned dictionary, whereas the values are the variable names
        of the pipeline.

        Right now, this only works for properties that don#t need any arguments,
        in pipelines such as "full_text". Others, such as "answers" return a function
        which needs arguments itself and can therefore not be used here.
        """

        properties = {a: getattr(self, a) for a in args}
        properties.update({v: getattr(self, k) for k, v in kwargs.items()})

        return deep_str_convert(properties)

    def non_interactive_x_funcs(self) -> dict[str, Extractor]:
        """return all non-interactive extractors"""
        return {k: v for k, v in self.x_funcs.items() if (not v._interactive)}

    def x_config_params(self, extract_name: str):
        # TODO: can we cache this somehow? Or re-calculate it when calling "config"?
        # get our required config parameters specifically for the function specified by "extract_name"
        config = self._x_config[self.pipeline_chooser].get(extract_name, {})
        config_params = {}
        for config_key in config:
            if config_key in self._config:
                v = self._config.get(config_key)
                config_params[config_key] = v
        return config_params

    def x(self, extract_name: str, *args, **kwargs):
        """
        call an extractor from our definition
        TODO: using *args and **kwargs the extractors parameters can be overriden
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
                    params = self.x_config_params(extract_name)
                    res = extractor_func._mapped_call(self, *args, config_params=params, **kwargs)
                    self._x_func_cache[key] = res
            else:
                params = self.x_config_params(extract_name)
                res = extractor_func._mapped_call(self, config_params=params, *args, **kwargs)

        except ExtractorException as e:
            logger.error(f"Extraction error in '{extract_name}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Extraction error in '{extract_name}': {e}")
            raise ExtractorException(f"could not get {extract_name} for {self}")
            # raise e

        return res[extract_name]

    def __getattr__(self, extract_name):
        """
        __getattr__ only gets called for non-existing variable names.
        So we can automatically avoid name collisions  here.

        >>> document.addresses

        instead of document.x['addresses']

        """
        return self.x(extract_name)

    def x_all(self):
        return {property: self.x(property) for property in self.x_funcs}

    def x_all_cached(self):
        return {self.x(property) for property in self.x_funcs}

    def run_all_extractors(self):
        """can be used for testing or pre-caching purposes"""
        # print(pdfdoc.elements)
        for x in self.non_interactive_x_funcs():
            self.x(x)

    def pre_cache(self):
        """in some situations, for example for caching purposes it would be nice
        to pre-cache all calculations this is done here by simply calling all functions..."""

        for x, ex in self.x_funcs.items():
            if ex._cache:
                self.x(x)

        return self

    # TODO: save document structure as a graph...
    # nx.write_graphml_lxml(G,'test.graphml')
    # nx.write_graphml(G,'test.graphml')

    def __repr__(self):
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
        return uuid.uuid4()

    def pipeline_graph(self, image_path: str | pathlib.Path = None, document_logic_id="current"):
        """
        Generate a visualization of the defined pipelines

        image_path:  file path for a generated image
        """
        # TODO: change into a static method
        graph = nx.DiGraph()
        if document_logic_id == "current":
            logic = self.x_funcs
        else:
            logic = self._x_funcs[document_logic_id]

        for name, f in logic.items():
            f_class = f.__class__.__name__ + "\n".join(f._out_mapping.keys())
            graph.add_node(f_class, shape="none")
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
