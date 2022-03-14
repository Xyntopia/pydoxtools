__version__ = '0.1.0'

import io
import logging
from pathlib import Path
from typing import Union, IO, List

from PIL import Image

from pydoxtools import pdf_utils, document


class LoadDocumentError(Exception):
    pass


logger = logging.getLogger(__name__)


def load_document(fobj: Union[str, Path, IO], source: str = "",
                  page_numbers: List[int] = None, maxpages: int = 0,
                  ocr: bool = False, ocr_lang="eng") -> document.Base:
    """takes a file-like object/string/path and returns a document corresponding to the
    filetype which can be used to extract data in a lazy fashion."""
    try:
        # TODO: add other document types
        # TODO: first try to check if the file is binary or string
        #       if string, chck if the path exists, if it doesn't take
        #       the "file" as a normal string/textdocument...
        if ocr:
            import pytesseract
            pdf = pytesseract.image_to_pdf_or_hocr(Image.open(fobj), extension='pdf', lang=ocr_lang)
            fobj = io.BytesIO(pdf)

        doc = pdf_utils.PDFDocument(fobj, source, page_numbers=page_numbers, maxpages=maxpages)
        return doc
    except:
        logger.exception("something went wrong!")
        raise LoadDocumentError(f"Could not open document {fobj}!")
