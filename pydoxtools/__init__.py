__version__ = '0.1.0'

import logging
from pathlib import Path
from typing import Union, IO

from pydoxtools import pdf_utils, document


class LoadDocumentError(Exception):
    pass


logger = logging.getLogger(__name__)


def load_document(fobj: Union[str, Path, IO]) -> document.Base:
    """takes a file-like object/string/path and returns a document corresponding to the
    filetype which can be used to extract data in a lazy fashion."""
    try:
        # TODO: add other document types
        # TODO: first try to check if the file is binary or string
        #       if string, chck if the path exists, if it doesn't take
        #       the "file" as a normal string/textdocument...
        doc = pdf_utils.PDFDocument(fobj)
        return doc
    except:
        logger.exception()
        raise LoadDocumentError(f"Could not open document {fobj}!")
