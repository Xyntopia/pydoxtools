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
            .pipe(fobj="_fobj").map("raw_content").cache(),
            pdf_utils.PDFFileLoader()
            .pipe(fobj="raw_content", page_numbers="_page_numbers", max_pages="_max_pages")
            .map("pages_bbox", "elements", "meta", "pages")
            .cache(),
            extract_textstructure.DocumentElementFilter(element_type=document.ElementType.Line)
            .pipe("elements").map("line_elements").cache(),
            extract_textstructure.TextBoxElementExtractor()
            .pipe("line_elements").map("text_box_elements").cache(),
            LambdaExtractor(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").map("text_box_list").cache(),
            LambdaExtractor(lambda tb: "\n\n".join(tb))
            .pipe(tb="text_box_list").map("full_text").cache()
        ],
        ".html": [
            HtmlExtractor()
            .pipe(raw_html="raw_content")
            .map("main_content_html", "keywords", "summary", "language", "goose_article",
                 "main_content")
        ],
        "*": [
            FileLoader(mode="auto")
            .pipe(fobj="_fobj", document_type="document_type", page_numbers="_page_numbers", max_pages="_max_pages")
            .map("raw_content").cache(),
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
    f_name = fobj.name
    try:
        # TODO: add other document types
        # TODO: first try to check if the file is binary or string
        #       if string, chck if the path exists, if it doesn't take
        #       the "file" as a normal string/textdocument...
        # TODO: move these functions into the document class itself...
        if ocr:
            import pytesseract
            if ocr_lang == "auto":
                pdf = pytesseract.image_to_pdf_or_hocr(Image.open(fobj), extension='pdf', lang=None)
                doc = pdf_utils.PDFDocumentOld(io.BytesIO(pdf), source, page_numbers=page_numbers, maxpages=maxpages)
                # TODO: if doc.lang is in ocr_langauge (for example when using: "eng+auto") then don't
                #       do anything.. as we already have the right document
                lang = doc.lang
                lang = pydoxtools.ocr_language_mappings.langdetect2tesseract.get(lang, None)
                fobj.seek(0)
            else:
                lang = ocr_lang
            pdf = pytesseract.image_to_pdf_or_hocr(Image.open(fobj), extension='pdf', lang=lang)
            fobj = io.BytesIO(pdf)

        doc = pdf_utils.PDFDocumentOld(
            fobj=fobj, source=source, page_numbers=page_numbers, maxpages=maxpages,
            model_size=model_size
        )
        return doc
    except pytesseract.pytesseract.TesseractError:
        logger.exception("tesseract problems")
        raise LoadDocumentError(f"language {ocr_lang} not supported!")
    except:
        logger.exception("something went wrong!")
        raise LoadDocumentError(f"Could not open document {f_name}!")
