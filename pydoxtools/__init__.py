__version__ = '0.1.0'

import io
import logging
from pathlib import Path
from typing import Union, IO, List

from PIL import Image

import pydoxtools.ocr_language_mappings
from pydoxtools import pdf_utils, document


class txtExtractor(document.Extractor):
    def __call__(self, fobj, page_numbers=None, max_pages=0):
        with open(fobj) as file:
            txt = file.read()
        return dict(txt=txt)


class Document(document.DocumentBase):
    _extractors = {
        ".pdf": [
            pdf_utils.PDFFileLoader()
            .pipe(fobj="_fobj", page_numbers="_page_numbers", max_pages="_max_pages")
            .map("pages_bbox", "elements", "meta", "pages")],
        ".html": [
            txtExtractor()
            .pipe(fobj="_fobj", page_numbers="_page_numbers", max_pages="_max_pages")
            .map(txt="raw_txt")]
    }

    # @functools.lru_cache
    def x(self, extract_name: str):
        """call an extractor from our definition"""
        extractor_logic: document.Extractor = self._x_funcs[extract_name]

        # TODO: add some caching logic to this call here!!
        return extractor_logic._mapped_call(self)


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
