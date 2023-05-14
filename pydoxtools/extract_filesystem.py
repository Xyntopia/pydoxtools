import functools
import pathlib
import typing
from pathlib import Path

import chardet

import pydoxtools.operators_base


class FileLoader(pydoxtools.operators_base.Operator):
    """Load data from path"""

    def __call__(
            self, fobj: bytes | str | Path | typing.IO, path=None,
            page_numbers=None, max_pages=None, document_type=None
    ) -> bytes | str:
        if path or isinstance(fobj, pathlib.Path):
            try:
                with open(fobj, "r") as file:
                    txt = file.read()
            except:
                with open(fobj, "rb") as file:
                    txt = file.read()
        elif isinstance(fobj, str | bytes):
            txt = fobj
        else:
            txt = fobj.read()

        if isinstance(txt, bytes):
            if document_type == "text/plain":
                if detected_encoding := chardet.detect(txt)['encoding']:
                    try:
                        txt = txt.decode(detected_encoding)
                    except UnicodeDecodeError:
                        txt = txt.decode("utf-8", errors='replace')

        # else:
        #    raise document_base.DocumentTypeError("Can not extract text from unknown document")
        return txt


def is_within_max_depth(path, directory, max_depth):
    depth = len(path.relative_to(directory).parts)
    return depth <= max_depth


class PathLoader(pydoxtools.operators_base.Operator):
    def __call__(
            self, directory: bytes | str | Path,
            # TODO: options: recursive-level
            exclude: list[str] = None
    ) -> list[Path]:
        @functools.cache
        def traverse_paths(max_depth: int=10, mode: str = "files"):
            """
            max_depth: maximum depth to which we should be able to list file paths...
            """

            # TODO:
            #   - exlude directories with min num of files
            #   - specify wildcard manually

            # dirs = file_utils.get_nested_paths(directory, "*", mode="dirs")
            # level = lambda path: len(path.relative_to(directory).parts)
            path_wildcard = "*"

            if mode == "files":
                path_list = [x for x in Path(directory).rglob(path_wildcard)
                             if (x.is_file() and is_within_max_depth(x, directory, max_depth))]
            elif mode == "dirs":
                path_list = [x for x in Path(directory).rglob(path_wildcard)
                             if (x.is_dir() and is_within_max_depth(x, directory, max_depth))]
            else:
                path_list = [x for x in Path(directory).rglob(path_wildcard)
                             if is_within_max_depth(x, directory, max_depth)]

            if exclude:
                # remove files from exclude list
                path_list: list[Path] = [
                    p for p in path_list
                    # make sure to exclude all strings from ignore list
                    if not len([i for i in exclude if i in str(p)]) > 0
                ]
            # it is usually not a good idea to use absolute paths for
            # information extraction purposes, as
            # the absolute paths obscure the names of the paths
            # which we can use to extract information from.
            # for example if a path was called "documentation"
            # path_list = [p.resolve() for p in path_list]
            return sorted(path_list)

        return traverse_paths
