import io
import pathlib
from pathlib import Path

from pydoxtools import document


class FileLoader(document.Extractor):
    def __init__(self, mode="auto"):
        super().__init__()
        self._mode = mode

    def __call__(
            self, fobj: bytes | str | Path | io.IOBase, document_type=None, page_numbers=None, max_pages=None
    ) -> bytes | str:
        if isinstance(fobj, str | bytes):
            txt = fobj
        elif isinstance(fobj, pathlib.Path):
            if self._mode == "auto":
                try:
                    with open(fobj, "r") as file:
                        txt = file.read()
                except:
                    with open(fobj, "rb") as file:
                        txt = file.read()
            else:
                with open(fobj, mode=self._mode) as file:
                    txt = file.read()
        else:
            txt = fobj.read()
        # else:
        #    raise document.DocumentTypeError("Can not extract text from unknown document")
        return txt
