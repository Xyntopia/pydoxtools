# -*- coding: utf-8 -*-
"""
@author: Thomas Meschede

This file implements common models used "all over the place" by componardo/comcharax.

"internal" models based on "pydantic" have an "underscore" as suffix:

Component_

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
from enum import Enum
from typing import List, Optional, Dict, Union, Set

import pandas as pd
from pydantic import BaseModel, Field, AnyHttpUrl

logger = logging.getLogger(__name__)


class Webpage(BaseModel):
    """
    Corresponds to scraped webpages
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


class DocumentData_(BaseModel):
    """
    All of the data that we can currently extract from a document
    such as a pdf, excel file, html page etc...
    """
    lang: str = None
    textboxes: List[str] = []
    urls: List[str] = []
    file: str = None
    images: List = []
    titles: List[str] = []
    docinfo: List[Dict[str, str]] = []
    tables: List[List] = []
    raw_content: List[str] = []
    keywords: List[str] = []
    url: str = None
    final_url: List[str] = []
    schemadata: Dict = {}  # schema.org data extracted from html meta tagss
    product_ids: Dict[str, str] = {}
    pdf_links: List[str] = []
    price: List[str] = []
