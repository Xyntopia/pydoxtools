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

from pydoxtools import models
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
    """Base class to build extraction logic for information extraction from
    unstructured documents and loading files"""

    # TODO:  how can we anhance the type checking for outputs?
    #        maybe turn this into a dataclass?

    def __init__(self):
        # try to keep __init__ with no arguments for Extractor..
        self._in_mapping: dict[str, str] = {}
        self._out_mapping: dict[str, str] = {}
        self._cache = False

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, typing.Any]:
        pass

    def _mapped_call(self, parent_document: "DocumentBase") -> dict[str, typing.Any]:
        # map objects from document properties to
        # processing function
        kwargs = {k: getattr(parent_document, v) for k, v in self._in_mapping.items()}
        output = self(**kwargs)
        if isinstance(output, dict):
            return {self._out_mapping[k]: v for k, v in output.items()}
        else:
            # use first key of out_mapping for output if
            # we only have a single return value
            return {next(iter(self._out_mapping)): output}

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
                uncombined_extractors: dict[str:dict[str:Extractor]] = {}
                extractor_combinations: dict[str:list[str]] = {}
                ex: Extractor | str
                for k, ex_list in new_class._extractors.items():
                    doc_type_x_funcs = {}
                    for ex in ex_list:
                        # strings indicate that we would like to
                        # add all the functions from that document type as well but with
                        # lower priority
                        if isinstance(ex, str):
                            extractor_combinations[k] = ex
                        else:
                            # go through all outputs of an extractor and
                            # map them o extraction variables inside document
                            # TODO: we could explicitly add the variables as property functions
                            #       which refer to the "x"-function in document?
                            for ex_key, ex_key_target in ex._out_mapping.items():
                                # input<->output mapping is already done i the extractor itself
                                # check out Extractor.pipe and Extractor.map member functions
                                doc_type_x_funcs[ex_key_target] = ex

                    uncombined_extractors[k] = doc_type_x_funcs

                # add all extrators by combining the different document types
                new_class._x_funcs = {}
                for k, v in uncombined_extractors.items():
                    # first take our other document type and then add the current document type
                    # itself on top of it because of its higher priority overwriting
                    # extractors of the lower priority extractors
                    # TODO: how do we make sure that we adhere to the tree structure?
                    #       we need to make sure that we generate the "lowest" priority (= top of tree)
                    #       document types first, and then subsequently until we are at the bottom
                    #       of the tree.

                    new_class._x_funcs[k] = {}
                    if base_type := extractor_combinations.get(k):
                        new_class._x_funcs[k].update(uncombined_extractors[base_type])
                    new_class._x_funcs[k].update(v)
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

    def __init__(
            self,
            fobj: str | bytes | Path | io.IOBase,
            source: str | Path = None,
            document_type: str = None,  # TODO: add "auto" for automatic recognition of the type using python-magic
            page_numbers: list[int] = None,
            max_pages: int = None
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
        TODO: we can calculate this in our metaclass as we have th pre-defined document
              types anyways.
        """
        return {**self._x_funcs.get("*", {}), **self._x_funcs.get(self.document_type, {})}

    # @functools.lru_cache
    def x(self, extract_name: str):
        """call an extractor from our definition"""
        extractor_func: Extractor = self.x_funcs[extract_name]

        # lru_cache currently has a memory leak so we 're not going to use it here
        # also as the function arguments to extractors won't change that much
        # we can use it in a similar way as "cached property"
        # TODO: what about "dynamic" extractors: for example a question/answering machine.
        #       we would also have to cache those variables...

        # we need to check for "is not None" as we also pandas dataframes in this
        # which cannot be checked for simple "is there"
        # check if we executed this function at some point...
        try:
            if not extractor_func._cache:
                res = extractor_func._mapped_call(self)
            elif (res := self._x_func_cache.get(extractor_func, None)) is not None:
                self._cache_hits += 1
            else:
                res = extractor_func._mapped_call(self)
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
    def extract(self) -> models.DocumentExtract:
        # TODO: return a datastructure which
        #       includes all the different extraction objects
        #       this datastructure should be serializable into
        #       json/yaml/xml etc...
        data = models.DocumentExtract.from_orm(self)
        return data

    @property
    def source(self) -> str:
        return self._source

    @property
    def fobj(self) -> str | io.IOBase:
        return self._fobj

    @property
    def final_url(self) -> list[str]:
        """sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document."""
        return []

    @property
    def parent(self) -> list[str]:
        """sources that embed this document in some way (for example as a link)
        (for example a product page which embeds
        a link to this document (e.g. a datasheet)
        """
        return []
