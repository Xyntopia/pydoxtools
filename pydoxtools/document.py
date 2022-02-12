from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Any, Union, BinaryIO

import langdetect
import pandas as pd
from pydoxtools import models, nlp_utils


class Base(ABC):
    """
    This class is the base for all document classes in pydoxtools and
    defines a common interface for all.

    This class also defines a basic extraction schema which derived
    classes can override
    """

    def __init__(
            self,
            fobj: Union[str, Path, BinaryIO],
            source: Union[str, Path]
            # Where does the extracted data come from? (Examples: URL, 'pdfupload', parent-URL, or a path)"
    ):
        self._fobj = fobj
        self._source = source

    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}({self._fobj},{self.source})>"

    @property
    def model(self) -> models.DocumentExtract:
        data = models.DocumentExtract.from_orm(self)
        return data

    # TODO: calculate md5-hash for the document and
    #       use __eq__ with that hash...
    #       we need this for caching purposes but also in order
    #       check if a document already exists...

    # TODO: test this for path, string, fobj and string path for different
    #       documents
    @property
    def filename(self) -> str:
        if isinstance(self._fobj, str):
            return str(self._fobj)
        elif isinstance(self._fobj, Path):
            return self._fobj.name
        else:
            return self._fobj.name

    @property
    def source(self) -> str:
        return self._source

    @property
    def fobj(self) -> Union[str, BinaryIO]:
        return self._fobj

    @property
    def type(self) -> str:
        """
        type such as "pdf", "html" etc...  can also be the mimetype!
        TODO: maybe we can do something generic here?
        """
        return "unknown"

    @property
    def list_lines(self):
        return []

    @property
    def tables(self) -> List[Dict[str, Dict[str, Any]]]:
        return []

    @property
    def tables_df(self) -> List["pd.DataFrame"]:
        return []

    @cached_property
    def lang(self) -> str:
        text = self.full_text.strip()
        if text:
            lang = langdetect.detect(text)
        else:
            lang = "unknown"
        return lang

    @property
    def textboxes(self) -> List[str]:
        return []

    @property
    def full_text(self) -> str:
        return ""

    @property
    def urls(self) -> List[str]:
        urls = nlp_utils.get_urls_from_text(self.full_text)
        return urls

    @property
    def images(self) -> List:
        return []

    @property
    def titles(self) -> List[str]:
        return []

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
    def keywords(self) -> List[str]:
        """a list of  keywords sometimes they are generated, other times
        they need to be extracted from the docment metadata"""
        return []

    @property
    def final_url(self) -> List[str]:
        """sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document."""
        return []

    @property
    def schemadata(self) -> Dict:
        """schema.org data extracted from html meta tags and other metainfos from documents

        TODO: more detailed description of return type"""
        return {}

    @property
    def product_ids(self) -> Dict[str, str]:
        return {}

    @property
    def pdf_links(self) -> List[str]:
        """sources that embed this document as a link (for example a product page which embeds
        a link to this document (e.g. a datasheet)

        TODO: rename to "parent_source" """
        return []

    @property
    def price(self) -> List[str]:
        """if prices are given in the document"""
        return []
