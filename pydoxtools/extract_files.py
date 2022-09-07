import pathlib
import io
from pathlib import Path

from pydoxtools import document


class TxtExtractor(document.Extractor):
    def __call__(self, fobj: bytes | str | Path | io.IOBase, document_type, page_numbers=None, max_pages=None):
        if isinstance(fobj, str | bytes):
            txt = fobj
        elif isinstance(fobj, pathlib.Path):
            with open(fobj) as file:
                txt = file.read()
        else:
            txt = fobj.read()
        # else:
        #    raise document.DocumentTypeError("Can not extract text from unknown document")
        return dict(txt=txt)
