# -*- coding: utf-8 -*-
"""
@author: Thomas Meschede

This file implements common models used "all over the place" by componardo/comcharax.

"internal" models based on "pydantic" have an "underscore" as suffix:

ComponentData

the corresponding DB model in neo4j is just called:

Component

without the underscore...

In the case of relationships inside a neomodel they are defined by either:

- Capital Letters in the beginning
- or an underscore at the end

properties (calculated & stored) are indicated by lower letters


"""

import datetime
import gzip
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, AnyHttpUrl

logger = logging.getLogger(__name__)


class Webpage(BaseModel):
    """
    Corresponds to scraped webpages

    TODO: get rid of this and have it as a standard "document"
    """
    url: AnyHttpUrl = Field(...)
    crawl_time: datetime.datetime = None
    embeddings_version: str = None
    embeddings: List[float] = None
    content_type: str = None
    history: AnyHttpUrl = None
    page_type: str = None
    html: str = Field(...)

    def compress(self):
        self.html = gzip.compress(self.html.encode('utf-8')) if self.html else None
        return self

    def decompress(self):
        self.html = gzip.decompress(self.html).decode('utf-8') if self.html else None
        return self


class DocumentExtract(BaseModel):
    """
    All of the data that we can currently extract from a document
    such as a pdf, excel file, html page etc...
    """
    uid: Optional[str]
    source: str = Field(
        ..., description="Where does the extracted data come from? (Examples: URL, 'pdfupload', parent-URL, or a path)"
    )
    type: str = Field(..., description="filetype of the data such as 'html' or 'pdf' or 'doc'")
    filename: Optional[str]

    lang: str = None
    textboxes: List[str] = []
    list_lines: List[str] = []
    urls: List[str] = []
    images: List = []
    titles: List[str] = []
    meta_infos: List[Dict[str, str]] = []
    # tables are row-wise!!  [{index -> {column -> value } }]
    # for a pandas dataframe we can export it like this:  df.to_dict('index')
    tables: List[Dict[str, Dict[str, Any]]] = []
    raw_content: List[str] = []
    keywords: List[str] = []
    url: str = None
    final_url: List[str] = []  # url extracted from the document itself where it points to "itself"
    schemadata: Dict = {}  # schema.org data extracted from html meta tags
    product_ids: Dict[str, str] = {}
    pdf_links: List[str] = []
    price: List[str] = []

    class Config:
        orm_mode = True
