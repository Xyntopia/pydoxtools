import pathlib
import typing
from pathlib import Path

from pydoxtools import document_base


class FileLoader(document_base.Operator):
    def __init__(self):
        super().__init__()

    def __call__(
            self, fobj: bytes | str | Path | typing.IO, document_type=None, page_numbers=None, max_pages=None
    ) -> bytes | str:
        if isinstance(fobj, str | bytes):
            txt = fobj
        elif isinstance(fobj, pathlib.Path):
            try:
                with open(fobj, "r") as file:
                    txt = file.read()
            except:
                with open(fobj, "rb") as file:
                    txt = file.read()
        else:
            txt = fobj.read()

        # TODO: don't just assume utf-8, but detect the actual encoding
        if isinstance(txt, bytes):
            try:
                txt = txt.decode("utf-8")
            except UnicodeDecodeError:
                pass

        # else:
        #    raise document_base.DocumentTypeError("Can not extract text from unknown document")
        return txt
