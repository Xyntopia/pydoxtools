import abc
import logging
import typing
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Union, BinaryIO, Any

import numpy as np
import pandas as pd
import spacy.tokens

from pydoxtools import models
from pydoxtools.cluster_utils import box_cols
from pydoxtools.extract_textstructure import _filter_boxes
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


class Extractor(ABC):
    """Base class to build extraction logic for information extraction from
    unstructured documents and loading files"""

    # TODO:  how can we anhance the type checking for outputs?
    #        maybe turn this into a dataclass?

    def __init__(self):
        self._in_mapping: dict[str, str] = {}
        self._out_mapping: dict[str, str] = {}

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, typing.Any]:
        pass

    def _mapped_call(self, parent_document: "DocumentBase") -> dict[str, typing.Any]:
        # map objects from document properties to
        # processing function
        kwargs = {k: getattr(parent_document, v) for k, v in self._in_mapping.items()}
        output = self(**kwargs)
        return {self._out_mapping[k]: v for k, v in output.items()}

    def pipe(self, *args, **kwargs):
        """
        configure input parameter mappings to this function

        keys: are the actual function parameters of the extractor function
        values: are the outside function names

        """
        self._in_mapping = kwargs
        self._in_mapping.update({k: k for k in args})
        return self

    def map(self, *args, **kwargs):
        """
        configure output parameter mappings to this function

        keys: are the
        """
        self._out_mapping = kwargs
        self._out_mapping.update({k: k for k in args})
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

    # in theory we could add additional arguments to this function which we could
    # pass in our documentbase class
    def __new__(cls, clsname, bases, attrs):
        # construct our class
        new_class: DocumentBase.__class__ = super(MetaDocumentClassConfiguration, cls).__new__(
            cls, clsname, bases, attrs)

        if hasattr(new_class, "_extractors"):
            if new_class._extractors:
                # TODO: add checks to make sure we don't have any name-collisions
                # configure class
                print(f"configure {new_class} class...")
                new_class._x_funcs = {}
                extractor: Extractor
                for k, ex_list in new_class._extractors.items():
                    doc_type_x_funcs = {}
                    for ex in ex_list:
                        for ex_key, ex_key_target in ex._out_mapping.items():
                            # input<->output mapping is already done i the extractor itself
                            # check out Extractor.pipe and Extractor.map member functions
                            doc_type_x_funcs[ex_key_target] = ex
                    new_class._x_funcs[k] = doc_type_x_funcs

                    # get all functions with their correct output and input
                    # func_map =
                    # new_class._extractor_map[k]
        else:
            raise ConfigurationError(f"no extractors defined in class {newclass}")

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
    _x_funcs: dict[str:dict[str, Extractor]] = {}

    def __init__(
            self,
            fobj: str | Path | BinaryIO,
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
        fobj: a file object which should be loaded. Can also be a string or a file path

        """
        self._fobj = fobj
        self._source = source
        self._document_type = document_type
        self._page_numbers = page_numbers
        self._max_pages = max_pages

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
            elif Path(self._fobj).exists() and not self._document_type:
                return Path(self._fobj).suffix
            else:
                raise DocumentTypeError(f"Could not find the document type for {self._fobj[-100:]} ...")
        except:
            try:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj} ...")
            except:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj[-100:]} ...")

    @cached_property
    def x_funcs(self):
        return self._x_funcs[self.document_type]

    # @functools.lru_cache
    def x(self, extract_name: str):
        """call an extractor from our definition"""
        extractor_func: Extractor = self.x_funcs[extract_name]

        # TODO: add some caching logic to this call here!!
        #       is it better to do this here or inside our Extractor itself?
        #       maybe inside the extractor... gives consistent variable names..
        #       we should also be able to specify it in the Extractor definition together
        #       with "pipe" and "map" maybe something lke "cache"
        #       what speaks against this is that we might want to have our caching
        #       document-related. So thats cached properties get deleted when the document
        #       itself is deleted...
        res = extractor_func._mapped_call(self)
        return res[extract_name]

    def __getattr__(self, extract_name):
        """
        __getattr__ only gets called for non-existing variable names.
        So we can automatically avoid name collisions  here.
        """
        return self.x(extract_name)

    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}({self._fobj},{self.source})>"

    # TODO: save document structure as a graph...
    # nx.write_graphml_lxml(G,'test.graphml')
    # nx.write_graphml(G,'test.graphml')

    @property
    def get_extract(self) -> models.DocumentExtract:
        # TODO: return a datastructure which
        #       includes all the different extraction objects
        #       this datastructure should be serializable into
        #       json/yaml/xml etc...
        data = models.DocumentExtract.from_orm(self)
        return data

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
    def fobj(self) -> Union[str, BinaryIO]:
        return self._fobj

    @property
    def mime_type(self) -> str:
        """
        type such as "pdf", "html" etc...  can also be the mimetype!
        TODO: maybe we can do something generic here?
        """
        return "unknown"

    @property
    def textboxes(self) -> List[str]:
        return []

    @cached_property
    def full_text(self) -> str:
        return ""

    @cached_property
    def pages(self) -> list[str]:
        """automatically divide text into approx. pages"""
        page_word_size = 500
        words = self.full_text.split()
        # for i in range(len(words)):
        pages = list(words[i:i + page_word_size] for i in range(0, len(words), page_word_size))
        return pages

    @cached_property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def docinfo(self) -> List[Dict[str, str]]:
        """list of document metadata such as author, creation date, organization"""
        return []

    @property
    def meta_infos(self) -> Dict:
        # specify metainfos in a better way
        return {}

    @property
    def raw_content(self) -> List[str]:
        """for example the raw html string in the case of an html document or the raw text for markdown"""
        return []

    @property
    def final_url(self) -> List[str]:
        """sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document."""
        return []

    @property
    def parent(self) -> List[str]:
        """sources that embed this document in some way (for example as a link)
        (for example a product page which embeds
        a link to this document (e.g. a datasheet)
        """
        return []

    @classmethod
    def pre_initialized(cls, fobj, **kwargs) -> "PDFDocumentOld":
        return cls(fobj, **kwargs).pre_cache()

    @classmethod
    @memory.cache
    def from_disk_cache(cls, fobj, **kwargs) -> "PDFDocumentOld":
        """return a pre -initialized document from disk_cache"""
        return repair_pdf_if_damaged(cls.pre_initialized)(fobj, **kwargs)
        # without repair:
        # return cls(fobj, table_extraction_params, page_numbers, max_pages).pre_cache()

    @classmethod
    @memory.cache
    def with_cached_elements(cls, fobj, **kwargs) -> "PDFDocumentOld":
        def extract(f):
            new = cls(f, **kwargs)
            cache = new.df_le, new.df_ge
            return new

        return repair_pdf_if_damaged(extract)(fobj)

    def pre_cache(self):
        """in some situations, for example for caching purposes it would be nice
        to pre-cache all calculations this is done here by simply calling all functions..."""
        # TODO: for some reason we can only use table_metrics right now for caching...
        #       try other things as well? at least this already gives us quiet a bit of
        #       time savings...
        res = (self.table_metrics, self.titles)
        # res = (
        #    self.pages, self.tables, self.table_metrics, self.side_content,
        #    self.textboxes, self.full_text, self.list_lines, self.main_content,
        #    self.titles, self.meta_infos, self.side_titles, self.pages_bbox
        # )
        # also cache all pages...
        # for p in self.pages:
        #    p.pre_cache()

        return self


class Page:
    def __init__(self, pagenum: int, doc: DocumentBase):
        self.pagenum = pagenum
        self.parent_document = doc

    @property
    def tbe(self):
        return self.parent_document.tbe

    def pre_cache(self):
        """in some situations, for example for caching purposes it would be nice
        to pre-cache all calculations this is done here by simply calling all functions..."""
        res = (
            self.tables, self.side_content,
            self.textboxes, self.full_text, self.main_content,
            self.titles, self.side_titles
        )
        return self

    @cached_property
    def page_bbox(self):
        # pikepdf:  p.mediabox
        return self.parent_document.pages_bbox[self.pagenum]

    @cached_property
    def area(self) -> float:
        return self.page_bbox[2] * self.page_bbox[3]

    @cached_property
    def df_ge_f(self) -> pd.DataFrame:
        """get filtered graph elements by throwing out a lot of unnecessary graphics
        elements which are probably not part of a table

        # TODO: parameterize this functions for hyperparameter tuning...

        we want to filter out a lot of graphics element which are not relevant. Those include:
        - very small elements
        - elements far away from text
        - elements which are are not a box or a vertical/horizontal line

        check out of we a lot of elements with "similar" x- or y coordinates in
        order to filter out "non-ordered" elements which
        we assume to be non-table graphical elments

        TODO: some of the following operations might be more efficient in the table-graph-search
        """

        if self.df_ge.empty:
            return pd.DataFrame()

        # TODO: filter specifically for each coordinate
        # TODO: filter for "unique-coordinate-density"
        #       as tables are ordered, we can assume that if we
        #       seach for "unique" coordinates, the density of
        #       tables should be even lower than that for figures.
        #       because of this we can theoretically improve the
        #       accuracy of this filter.

        # TODO: use these as hyperparameters
        min_size = 5.0  # minimum size of a graphics element
        margin = 20  # margin of the page
        max_area_page_ratio = 0.4  # maximum area on a page to occupy by a graphics element

        # get minimum length for lines by searching for
        # the minimum height/width of a text box
        # we do this, because we assume that graphical elements should be at least this
        # big in order to be part of a table
        min_elem_x = max(self.txt_box_df.w.min(), min_size)
        min_elem_y = max(self.txt_box_df.h.min(), min_size)

        # check if the following calculations are duplicated anywhere...
        ge = self.df_ge
        ge['w'] = ge.x1 - ge.x0
        ge['h'] = ge.y1 - ge.y0
        ge['area'] = ge.w * ge.h
        ge['area_ratio'] = (ge.area / self.area)
        # df_ge['w+h'].hist(bins=100)
        # filter out all elements that occupy more than max_area_page_ratio of page space
        ge_v = ge[ge.area_ratio < max_area_page_ratio]
        # filter out all elements that are thinner than min_elem_x- and y
        # under certain conditions
        ge_v = ge_v.query('((w>h) and (w>@min_elem_x)) or ((h>w) and (h>@min_elem_y))').copy()
        # elements that are "outside the page area/margin should be discarded
        # "standard" margin
        ge_v = ge_v[((ge_v[box_cols[2:]] + margin) < (self.page_bbox[2:])).all(1)
                    & ((ge_v[box_cols[:2]] - margin) > (self.page_bbox[:2])).all(1)]
        # use the smallest rectangle which encloses textboxes plus a small margin
        # min_x = self.box_groups.x0.min()
        # max_x = self.box_groups.x1.max()
        # min_y = self.box_groups.y0.min()
        # max_y = self.box_groups.y1.max()
        # ge_v = ge_v[((ge_v.x0 + margin) > min_x) & ((ge_v.x1 - margin) < max_x)
        #             & ((ge_v.y0 + margin) > min_y) & ((ge_v.y1 - margin) < max_y)]

        # elements

        return ge_v

    @cached_property
    def df_le(self) -> pd.DataFrame:
        """line elements of page"""
        if not self.parent_document.df_le.empty:
            return self.parent_document.df_le.loc[self.parent_document.df_le.p_id == self.pagenum].copy()
        else:
            return pd.DataFrame()  # empty dataframe

    @cached_property
    def df_ge(self) -> pd.DataFrame:
        """graphic elements of page"""
        if not self.parent_document.df_ge.empty:
            return self.parent_document.df_ge[
                self.parent_document.df_ge.p_id == self.pagenum].copy()
        else:
            return pd.DataFrame()  # empty dataframe

    @property
    def tables(self) -> typing.List["PDFTable"]:
        if self.df_le.empty:
            return []
        return [t for t in self.table_candidates if t.is_valid]

    @property
    def table_areas(self) -> pd.DataFrame:
        return pd.DataFrame([t.bbox for t in self.tables])

    @property
    def distance_threshold(self) -> float:
        # we keep this number constant as the same effect can be gained
        # through tbe.area_detection_params but a lot more fine grained as
        # it directly controls the sensitivity of the distance function
        return 10.0  # merge everything with a distance of less than 10..

    def detect_table_area_candidates(self):
        """
        Detect tables from elements sucha s textboxes & graphical elements.
        the function expects a range of parameters which need to be tuned.

        TODO: sort out non-table area regions after every iteration and speed up subsequent
              table search iterations this way.. But optimize this on a recall-basis
              in order to make sure we don't sort out any valid tables...
        """
        boxes = pd.concat([
            self.df_le[box_cols] if ("t" in self.tbe.extraction_method and (not self.df_le.empty)) else pd.DataFrame(),
            self.df_ge_f[box_cols] if (
                    "g" in self.tbe.extraction_method and (not self.df_ge_f.empty)) else pd.DataFrame(),
        ])
        if len(boxes) == 0:
            return pd.DataFrame(), []

        box_levels = []

        # TODO: if (graphic) boxes are empty, revert to text-based...
        # TODO: do this in several (configurable) iterations
        if not self.tbe.area_detection_distance_func_params:
            raise ValueError("no area_detection_distance_func_params defined!")
        if len(boxes) > 1:  # merge boxes to table areas..
            for level, param_level in enumerate(self.tbe.area_detection_distance_func_params):
                x = gu.calc_pairwise_matrix(
                    gu.pairwise_weighted_distance_combination, boxes.values, diag=0,
                    parameter_list=param_level
                )

                boxes["groups"], dist_m = gu.distance_cluster(
                    distance_matrix=x, distance_threshold=self.distance_threshold
                )
                # create new column with the type of group (hb,hm,ht,vb,vm,vt) and their labels
                boxes = gu.merge_bbox_groups(boxes, "groups")
                box_levels.append(boxes)
                if len(boxes) < 2:
                    break

        # TODO: check variance to sort out "bad" columns as an additional parameter?
        #       but maybe it would also be a good idea to simply do that during the distance calculation
        # line_groups.groupby('vh_left_top_group').x0.apply(lambda x: x.var())

        # filter our empty groups
        # TODO: right now, we don't really know what would be a good filter...
        #       maybe do this by using an optimization approach
        table_groups = _filter_boxes(
            boxes,
            min_area=self.tbe.min_table_area,
            min_aspect_ratio=self.tbe.min_aspect_ratio,
            max_aspect_ratio=self.tbe.max_aspect_ratio
        )

        # sort table candidates according to y-coordinates top-to-bottom
        table_groups = table_groups.sort_values(
            by=["y1", "x0", "y0", "x1"], ascending=[False, True, False, True])

        return table_groups, box_levels

    @property
    def table_candidates(self) -> typing.List["PDFTable"]:
        tables = [PDFTable(parent_page=self, initial_area=row[box_cols]) for _, row in
                  self.detect_table_area_candidates()[0].iterrows()]
        # we only have a valid table if there is actualy text to be processed...
        # TODO:  also handly tables with figures only at some point in the future?
        # TODO: should we sort out "bad" areas here already? may speed up table extraction...
        return [t for t in tables if not t.df_le.empty]

    @property
    def table_candidate_boxes_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.bbox for t in self.table_candidates])


def DocumentClassFactory():
    """
    TODO: we could provide "fast" configurations or also simply custom configurations etc...
    """
    extractors = {
        "pdf": "test"
    }

    class NewDocument():
        __extractors = extractors

    return NewDocument
