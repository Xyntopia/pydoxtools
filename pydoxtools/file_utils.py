from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import hashlib
import typing
from pathlib import Path
from urllib.parse import urlparse


def generate_unique_pdf_filename_from_url(source_url: str) -> str:
    """
    This function generates a unique filename by appending two digits from an md5-hash
    of the source of a document in order to discern between documents with the same filename
    """
    file_id = hashlib.md5(source_url.encode('utf-8')).hexdigest()[-2:]
    parsed = urlparse(source_url)
    if parsed.path[-3:].lower() == "pdf":
        fn = parsed.path.split('/')[-1]
        return fn[:-3] + f"{file_id}.pdf"
    else:
        raise NotImplementedError(
            f"{source_url}: we can not use this function for documents with non-pdf endings yet...")


def get_nested_paths(directory: typing.Union[str, Path], path_wildcard: str = "*", mode="files") -> [Path]:
    """
    get all pdf files in subdirectory
    :param file_ending:
    :return:
    """
    if mode == "files":
        files = [x for x in Path(directory).rglob(path_wildcard) if x.is_file()]
    elif mode == "dirs":
        files = [x for x in Path(directory).rglob(path_wildcard) if x.is_dir()]
    else:
        files = [x for x in Path(directory).rglob(path_wildcard)]
    return files
