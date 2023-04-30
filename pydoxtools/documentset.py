from pathlib import Path

import pydoxtools.operators
from pydoxtools import document_base
from pydoxtools import file_utils


class DirectoryLoader(pydoxtools.operators.Operator):
    def __init__(self):
        super().__init__()

    def __call__(
            self, directory: bytes | str | Path,
            # TODO: options: recursive-level
            exclude: list[str] = None
    ) -> list[Path]:
        root_dir = "../pydoxtools"
        dirs = file_utils.get_nested_paths(directory, "*", mode="dirs")
        level = lambda path: len(path.relative_to(root_dir).parts)

        path_list = [
            p for p in file_utils.get_nested_paths(root_dir, "*")
            if not (
                    len([i for i in exclude if i in str(p)]) > 0
                    or p.is_dir()
            )
        ]

        return path_list


class DocumentSet(document_base.Pipeline):
    """
    This class loads an entire set of documents and processes
    it using a pipeline.
    """
    _operators = {
        "*": [
            # TODO:  add a string filter which can be used to filte paths & db entries
            #        and is simply a bit more generalized ;)
            DirectoryLoader()
            .pipe(directory="directory")
            .out("path_list")
            .config(exclude=[
                '.git/', '.idea/', '/node_modules', '/dist',
                '/__pycache__/', '.pytest_cache/'
            ]).cache()
        ]
    }

    def __init__(self, directory):
        self._directory = directory

    @property
    def directory(self):
        return self._directory


# testing directory loading
root_dir = "../pydoxtools"
