from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Union, BinaryIO

import numpy as np
import spacy.tokens

from pydoxtools import models


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


class DocumentLoader(ABC):
    """Base class for loading documents from different kinds of files"""
    pass


class Extractor(ABC):
    """Base class to build extraction logic for information extraction from
    unstructured documents"""
    pass


class Document:
    """
    This class is the base for all document classes in pydoxtools and
    defines a common interface for all.

    This class also defines a basic extraction schema which derived
    classes can override
    """
    __loader: list[DocumentLoader] = []
    __extractor: list[DocumentLoader] = []

    @classmethod
    def add_loader(cls, new_loader: DocumentLoader):
        cls.__loader.append(new_loader)

    @classmethod
    def add_extractor(cls):
        cls.__extractor.append()

    def __init__(
            self,
            fobj: Union[str, Path, BinaryIO],
            source: Union[str, Path],
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

    def __repr__(self):
        return f"{self.__module__}.({self._fobj},{self.source})>"

    @property
    def type(self):
        return 'unknown'

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
