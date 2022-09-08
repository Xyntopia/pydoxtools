import abc
import io
import logging
import typing
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import List, Any

import numpy as np
import spacy.tokens

from pydoxtools.settings import settings

logger = logging.getLogger(__name__)

memory = settings.get_memory_cache()


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
        self._cache = False
        self._dynamic_config: dict[str, str] = {}

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, typing.Any]:
        pass

    def _mapped_call(self, parent_document: "DocumentBase", config: dict[str, Any] = None) -> dict[str, typing.Any]:
        # map objects from document properties to
        # processing function
        kwargs = {}
        for k, v in self._in_mapping.items():
            # first check if parameter is available as an extractor
            if v in parent_document.x_funcs:
                kwargs[k] = parent_document.x(v)
            else:  # get "native" member-variables if not found an extractor with that name
                kwargs[k] = getattr(parent_document, v)
        # get potential configuration parameters to override function call
        if config:
            override_parameters = {self._dynamic_config[k]: v for k, v in config.items()}
            kwargs.update(override_parameters)

        output = self(**kwargs)
        if isinstance(output, dict):
            return {self._out_mapping[k]: v for k, v in output.items()}
        else:
            # use first key of out_mapping for output if
            # we only have a single return value
            return {next(iter(self._out_mapping)): output}

    def config(self, *args, **kwargs):
        """
        dynamically configure the extractor.

        This function will be passed through to doc.x.config() in a document
        instance in order to make some extractor arguments dynamic and changeable.
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
        self._out_mapping = kwargs
        self._out_mapping.update({k: k for k in args})
        return self

    def cache(self):
        """indicate to document that we want this extractor to be cached"""
        self._cache = True
        return self


class ConfigurationError(Exception):
    pass


class DocumentTypeError(Exception):
    pass


class MetaDocumentClassConfiguration(type):
    """
    configures derived document classes on construction.

    ALso checks Extractors etc...  for consistency
    """

    # TODO: we can probably refactor this function to make it easir to understand
    #       decouple etc...

    # in theory, we could add additional arguments to this function which we could
    # pass in our documentbase class
    def __new__(cls, clsname, bases, attrs):
        # construct our class
        new_class: DocumentBase.__class__ = super(MetaDocumentClassConfiguration, cls).__new__(
            cls, clsname, bases, attrs)

        if hasattr(new_class, "_extractors"):
            if new_class._extractors:
                # TODO: add checks to make sure we don't have any name-collisions
                # configure class
                logger.info(f"configure {new_class} class...")
                uncombined_extractors: dict[str, dict[str, Extractor]] = {}
                extractor_combinations: dict[str, list[str]] = {}
                uncombined_x_configs: dict[str, dict[str, list[str]]] = {}
                ex: Extractor | str
                for doc_type, ex_list in new_class._extractors.items():
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
                            # map them o extraction variables inside document
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

                    uncombined_extractors[doc_type] = doc_type_x_funcs
                    uncombined_x_configs[doc_type] = doc_type_x_config

                # add all extrators by combining the different document types
                new_class._x_funcs = {}
                new_class._x_config = {}
                doc_type: str
                for doc_type in uncombined_extractors:
                    # first take our other document type and then add the current document type
                    # itself on top of it because of its higher priority overwriting
                    # extractors of the lower priority extractors
                    # TODO: how do we make sure that we adhere to the tree structure?
                    #       we need to make sure that we generate the "lowest" priority (= top of tree)
                    #       document types first, and then subsequently until we are at the bottom
                    #       of the tree.

                    # TODO: add classes recursivly
                    new_class._x_funcs[doc_type] = {}
                    new_class._x_config[doc_type] = {}

                    # build class combination in correct order:
                    # the first one is the least important
                    doc_type_order = ["*"] + list(
                        reversed(extractor_combinations[doc_type])) + [doc_type]

                    for ordered_doc_type in doc_type_order:
                        # add extractors from a potential base document
                        new_class._x_funcs[doc_type].update(uncombined_extractors[ordered_doc_type])
                        new_class._x_config[doc_type].update(uncombined_x_configs[ordered_doc_type])

        else:
            raise ConfigurationError(f"no extractors defined in class {new_class}")

        return new_class


class DocumentBase(metaclass=MetaDocumentClassConfiguration):
    """
    This class is the base for all document classes in pydoxtools and
    defines a common interface for all.

    This class also defines a basic extraction schema which derived
    classes can override
    """

    # TODO: use pandera (https://github.com/unionai-oss/pandera)
    #       in order to validate dataframes exchanged between extractors & loaders
    #       https://pandera.readthedocs.io/en/stable/pydantic_integration.html

    # TODO: how do we change extraction configuration "on-the-fly" if we have
    #       for example a structured dcument vs unstructered (PDF: unstructure,
    #       Markdown: structured)
    #       in this case table extraction algorithms for example would have to
    #       behave differently. We would like to use
    #       a different extractor configuration in that case...
    #       in other words: each extractor needs to be "conditional"

    _extractors: dict[str, list[Extractor]] = {}
    # sorts for all extractor variables..
    _x_funcs: dict[str, dict[str, Extractor]] = {}
    # doctype-dict of x-out var dict of function configuration parameter dicts:
    _x_config: dict[str, dict[str, dict[str, Any]]] = {}

    def __init__(
            self,
            fobj: str | bytes | Path | io.IOBase,
            source: str | Path,
            document_type: str,  # TODO: add "auto" for automatic recognition of the type using python-magic
            page_numbers: list[int],
            max_pages: int,
            config: dict[str, Any]
    ):
        """
        ner model:

        if a "spacy_model" was specified use that.
        else if "model_size" was specified, use generic spacy language model
        else  use generic, multilingual ner model "xx_ent_wiki_sm"

        source: Where does the extracted data come from? (Examples: URL, 'pdfupload', parent-URL, or a path)"
        fobj: a file object which should be loaded.
            - if it is a string or bytes object:   the string itself is the document!
            - if it is a pathlib.Path: load the document from the path
            - if it is a file object: load document from file object (or bytestream  etc...)

        """
        self._fobj = fobj
        self._source = source
        self._document_type = document_type
        self._page_numbers = page_numbers
        self._max_pages = max_pages
        self._cache_hits = 0
        self._x_func_cache: dict[Extractor, dict[str, Any]] = {}
        self._config = config

    @cached_property
    def document_type(self):
        """
        detect doc type based on file-ending
        TODO add a doc-type extractor using for example python-magic
        """
        try:
            if self._document_type:
                return self._document_type
            elif hasattr(self._fobj, "name"):
                return Path(self._fobj.name).suffix
            # get type from path suffix
            elif self._fobj.exists() and not self._document_type:
                return self._fobj.suffix
            else:  # for example if it is a string without a type
                # TODO: detect type with python-magic here...
                raise DocumentTypeError(f"Could not find the document type for {self._fobj[-100:]} ...")
        except:
            try:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj} ...")
            except:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj[-100:]} ...")

    @cached_property
    def x_funcs(self) -> dict[str, Extractor]:
        """
        get all extractors and their proprety names
        """
        return self._x_funcs.get(self.document_type, {})

    def x_config_params(self, extract_name: str):
        # TODO: can we cache this somehow? Or re-calculate it when calling "config"?
        config = self._x_config[self.document_type].get(extract_name, {})
        config_params = {k: self._config[k] for k in config}
        return config_params

    # @functools.lru_cache
    def x(self, extract_name: str):
        """call an extractor from our definition"""
        extractor_func: Extractor = self.x_funcs[extract_name]

        # we need to check for "is not None" as we also pandas dataframes in this
        # which cannot be checked for simple "is there"
        # check if we executed this function at some point...
        try:
            if not extractor_func._cache:
                config_params = self.x_config_params(extract_name)
                res = extractor_func._mapped_call(self, config_params)
            elif (res := self._x_func_cache.get(extractor_func, None)) is not None:
                self._cache_hits += 1
            else:
                config_params = self.x_config_params(extract_name)
                res = extractor_func._mapped_call(self, config_params)
                self._x_func_cache[extractor_func] = res
        except:
            logger.exception(f"problem with extractor {extract_name}")
            raise ExtractorException(f"could not extract {extract_name} from {self} using {extractor_func}!")

        return res[extract_name]

    def __getattr__(self, extract_name):
        """
        __getattr__ only gets called for non-existing variable names.
        So we can automatically avoid name collisions  here.
        """
        return self.x(extract_name)

    def x_all(self):
        return {property: self.x(property) for property in self.x_funcs}

    def x_all_cached(self):
        return {self.x(property) for property in self.x_funcs}

    def run_all_extractors(self):
        """can be used for testing purposes"""
        # print(pdfdoc.elements)
        for x in self.x_funcs:
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

    @property
    def source(self) -> str:
        return self._source

    @property
    def fobj(self):
        return self._fobj

    @cached_property
    def filename(self) -> str | None:
        if hasattr(self._fobj, "name"):
            return self._fobj.name
        elif isinstance(self._fobj, Path):
            return self._fobj.name
        else:
            return None

    """
    @property
    def final_url(self) -> list[str]:
        ""sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document.""
        return []

    @property
    def parent(self) -> list[str]:
        ""sources that embed this document in some way (for example as a link)
        (for example a product page which embeds
        a link to this document (e.g. a datasheet)
        ""
        return []
    """
