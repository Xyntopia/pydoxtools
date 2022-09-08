__version__ = '0.5.0'

import io
import logging
from pathlib import Path
from typing import Any

from . import document
from . import extract_textstructure
from . import pdf_utils
from .extract_files import FileLoader
from .extract_html import HtmlExtractor
from .extract_logic import LambdaExtractor

logger = logging.getLogger(__name__)


class Document(document.DocumentBase):
    """
    A standard document logic configuration which should work
    on most documents.

    In order to declare a different logic it is best to take this logic here as a
    starting point.

    It is possible to exchange individual modules such as HTML extractors etc..

    One can also change the configuration of individual extractors. For example
    of the Table Extractor or Space models...
    """
    _extractors = {
        ".pdf": [
            FileLoader(mode="rb")  # pdfs are usually in binary format...
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

    def __init__(
            self,
            fobj: str | bytes | Path | io.IOBase,
            source: str | Path = None,
            document_type: str = None,  # TODO: add "auto" for automatic recognition of the type using python-magic
            page_numbers: list[int] = None,
            max_pages: int = None,
            config: dict[str, Any] = None):
        """Standard Document class adds some default variables..."""
        super().__init__(fobj, source, document_type, page_numbers, max_pages, config)
