from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import pathlib
import typing
from pathlib import Path

import chardet

import pydoxtools.operators_base


def force_decode(txt: bytes | str):
    if isinstance(txt, bytes):
        if detected_encoding := chardet.detect(txt)['encoding']:
            try:
                return txt.decode(detected_encoding)
            except UnicodeDecodeError:
                pass
        return txt.decode("utf-8", errors='replace')
    else:
        return txt


def load_raw_file_content(fobj: bytes | str | Path | typing.IO) -> bytes | str:
    if isinstance(fobj, pathlib.Path):
        try:
            with open(fobj) as file:
                txt = file.read()
        except:
            with open(fobj, "rb") as file:
                txt = file.read()
    elif isinstance(fobj, str | bytes):
        txt = fobj
    else:
        txt = fobj.read()

    return txt


def is_within_max_depth(path, directory, max_depth):
    depth = len(path.relative_to(directory).parts)
    return depth <= max_depth


class PathLoader(pydoxtools.operators_base.Operator):
    def __call__(
            self, directory: bytes | str | Path,
            # TODO: options: recursive-level
            exclude: list[str] = None
    ) -> typing.Callable:
        """
        Loads a list of paths from the given directory with the specified constraints.

        Args:
            directory (Union[bytes, str, Path]): The directory where the paths are to be fetched.
            exclude (List[str], optional): List of filenames or directories to exclude from the result. Defaults to None.

        Returns:
            Return a function which traverses the specified directory
        """

        @functools.cache
        def traverse_paths(max_depth: int = 10, mode: str = "", wildcard: str = "*"):
            """
            Travels through the directory and returns a list of paths according to the mode and max_depth.

            Args:
                max_depth (int, optional): Maximum depth to which paths should be listed. Defaults to 10.
                mode (str, optional): Mode to determine which paths are returned ("files", "dirs", or ""). Defaults to "".
                wildcard (str, optional): Wildcard to filter files..

            Returns:
                List[Path]: List of paths in the directory according to the specified mode and max_depth.
            """

            # TODO:
            #   - exlude directories with min num of files
            #   - specify wildcard manually

            # dirs = file_utils.get_nested_paths(directory, "*", mode="dirs")
            # level = lambda path: len(path.relative_to(directory).parts)

            if mode == "files":
                path_list = [x for x in Path(directory).rglob(wildcard)
                             if (x.is_file() and is_within_max_depth(x, directory, max_depth))]
            elif mode == "dirs":
                path_list = [x for x in Path(directory).rglob(wildcard)
                             if (x.is_dir() and is_within_max_depth(x, directory, max_depth))]
            else:
                path_list = [x for x in Path(directory).rglob(wildcard)
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
