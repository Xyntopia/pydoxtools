from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import mimetypes
import pathlib

from pydoxtools import Document, DocumentBag
from pydoxtools.settings import _PYDOXTOOLS_DIR

test_dir_path = pathlib.Path(__file__).parent.absolute()


def make_path_absolute(f: pathlib.Path | str):
    return test_dir_path / pathlib.Path(f)


def test_pipeline_graph():
    # TODO: generate graphs for all document types
    for k in Document._pipelines:
        ending = mimetypes.guess_extension(k, strict=False) or k
        ending = ending.replace("/", "_")
        Document.pipeline_graph(
            image_path=_PYDOXTOOLS_DIR / f"docs/images/document_logic_{ending}.svg",
            document_logic_id=k
        )

    for k in DocumentBag._pipelines:
        DocumentBag.pipeline_graph(
            image_path=_PYDOXTOOLS_DIR / f"docs/images/documentbag_logic_{k}.svg",
            document_logic_id=k
        )


def test_documentation_generation():
    docs = Document.markdown_docs()
    Document_docs = f"""
# [pydoxtools.Document][]

::: pydoxtools.Document

## Text extraction attributes and functions

The [pydoxtools.Document][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

{docs}
""".strip()
    with open(make_path_absolute('../docs/document.md'), "w") as f:
        f.write(Document_docs)

    ###  and now DocumentBag

    docbag = DocumentBag.markdown_docs()

    Document_docs = f"""
# [pydoxtools.DocumentBag][]

::: pydoxtools.DocumentBag

## Text extraction attributes and functions

The [pydoxtools.DocumentBag][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).


{docbag}
    """.strip()
    with open(make_path_absolute('../docs/documentbag.md'), "w") as f:
        f.write(Document_docs)


if __name__ == "__main__":
    test_pipeline_graph()
    test_documentation_generation()
