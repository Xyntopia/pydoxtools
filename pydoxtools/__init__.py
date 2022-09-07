__version__ = '0.1.0'

import io
import logging
from pathlib import Path
from typing import Union, IO, List
import pandas as pd

from PIL import Image

import pydoxtools.ocr_language_mappings
from pydoxtools import extract_textstructure
from pydoxtools import pdf_utils, document
from pydoxtools.extract_files import FileLoader
from pydoxtools.extract_html import HtmlExtractor
from pydoxtools.extract_logic import LambdaExtractor


class Document(document.DocumentBase):
    """
    A standard document logic configuration which should work
    on most documents
    """
    _extractors = {
        ".pdf": [
            FileLoader(mode="rb") # pdfs are usually in binary format...
            .pipe(fobj="_fobj").out("raw_content").cache(),
            pdf_utils.PDFFileLoader()
            .pipe(fobj="raw_content", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("pages_bbox", "elements", "meta", "pages")
            .cache(),
            extract_textstructure.DocumentElementFilter(element_type=document.ElementType.Line)
            .pipe("elements").out("line_elements").cache(),
            extract_textstructure.TextBoxElementExtractor()
            .pipe("line_elements").out("text_box_elements").cache(),
            LambdaExtractor(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").out("text_box_list").cache(),
            LambdaExtractor(lambda tb: "\n\n".join(tb))
            .pipe(tb="text_box_list").out("full_text").cache()
        ],
        ".html": [
            HtmlExtractor()
            .pipe(raw_html="raw_content")
            .out("main_content_html", "keywords", "summary", "language", "goose_article",
                 "main_content")
        ],
        "*": [
            FileLoader(mode="auto")
            .pipe(fobj="_fobj", document_type="document_type", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("raw_content").cache(),
        ]
    }


class LoadDocumentError(Exception):
    pass


logger = logging.getLogger(__name__)


def load_document(fobj: Union[str, Path, IO], source: str = "",
                  page_numbers: List[int] = None, maxpages: int = 0,
                  ocr: bool = False, ocr_lang="eng", model_size="") -> document.DocumentBase:
    """takes a file-like object/string/path and returns a document corresponding to the
    filetype which can be used to extract data in a lazy fashion."""
    pass