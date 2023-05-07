import pathlib
import typing
from pathlib import Path

import pydoxtools.operators_base


class FileLoader(pydoxtools.document_base.Operator):
    """Load data from path"""

    def __call__(
            self, fobj: bytes | str | Path | typing.IO, path=None, page_numbers=None, max_pages=None
    ) -> bytes | str:
        if path:
            with open(path, "r") as file:
                txt = file.read()
        elif isinstance(fobj, str | bytes):
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


class PathLoader(pydoxtools.document_base.Operator):
    def __init__(self, mode: str = "files"):
        super().__init__()
        self._mode = mode

    def __call__(
            self, directory: bytes | str | Path,
            # TODO: options: recursive-level
            exclude: list[str] = None
    ) -> list[Path]:
        # dirs = file_utils.get_nested_paths(directory, "*", mode="dirs")
        # level = lambda path: len(path.relative_to(directory).parts)
        path_wildcard = "*"

        if self._mode == "files":
            path_list = [x for x in Path(directory).rglob(path_wildcard) if x.is_file()]
        elif self._mode == "dirs":
            path_list = [x for x in Path(directory).rglob(path_wildcard) if x.is_dir()]
        else:
            path_list = [x for x in Path(directory).rglob(path_wildcard)]

        if exclude:
            # remove files from exclude list
            path_list: list[Path] = [
                p for p in path_list
                # make sure to exclude all strings from ignore list
                if not len([i for i in exclude if i in str(p)]) > 0
            ]

        # make paths absolute
        # it is usually not a good idea to use absolute paths, as
        # the absolute paths obscure the names of the paths
        # which we can use to extract information from.
        # for example if a path was called "documentation"
        # path_list = [p.resolve() for p in path_list]

        return sorted(path_list)
