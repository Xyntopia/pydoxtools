from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import mimetypes

import pandas as pd
from packaging import version

import pydoxtools.operators_base

logger = logging.getLogger(__name__)

import pandoc

pydoxtools_link = "https://github.com/jgm/pandoc/releases/tag/2.19.2"

try:
    import pandoc.types

    pandoc_version = pandoc._configuration['version']
    if version.parse(pandoc_version) < version.parse('2.14.2'):
        logger.warning(f"installed pandoc version {pandoc_version}, which doesn't support rtf file format!"
                       f"in order to be able to use rtf, you need to install a pandoc version >= 2.14.2"
                       f"Checkout this link here: {pydoxtools_link}")
    pandoc_installed = True

except RuntimeError:
    logger.warning(f"""pandoc does not seem to be installed, in order to load some documents
    such as *.docx, *.rtf this library needs to be installed.
    
    Checkout this link here: {pydoxtools_link}
    """)

    pandoc_installed = False


def extract_list(elt):
    output = []
    for e in elt:
        if isinstance(e, pandoc.types.BulletList):
            output.append(extract_list(e[0]))
        elif isinstance(e, pandoc.types.OrderedList):
            output.append(extract_list(e[1]))
        elif isinstance(e, list):
            out = extract_list(e)
            if isinstance(out, str):
                output.append(out)
            elif len(out) > 1:
                output.append(out)
            else:
                output.extend(out)
        elif isinstance(e, tuple):
            # handles formatting of numbered lists
            continue
        else:
            output.append(pandoc.write(e, format="markdown").strip())

    if isinstance(output, list) and len(output) == 1:
        return output[0]
    else:
        return output


class PandocLoader(pydoxtools.operators_base.Operator):
    """
    Converts a string or a raw byte string into pandoc intermediate format.
    """

    def __init__(self):
        super().__init__()

    def __call__(
            self, raw_content: bytes | str, document_type: str
    ) -> "pandoc.types.Pandoc":
        if not pandoc_installed:
            raise RuntimeError("""Pandoc files can not be loaded, as pandoc is not installed""")
        if ext := mimetypes.guess_extension(document_type):
            ext = ext.strip(".")
        else:
            ext = document_type
        type_mapping = {
            "md": "markdown",
            'text/rtf': "rtf"
        }
        pandoc_format = pandoc.read(
            raw_content,
            format=type_mapping.get(ext, ext)
        )
        return pandoc_format


class PandocBlocks(pydoxtools.operators_base.Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, pandoc_document: "pandoc.types.Pandoc") -> list["pandoc.types.Block"]:
        txtblocks = [elt for elt in pandoc_document[1] if isinstance(elt, pandoc.types.Block)]
        return txtblocks


class PandocConverter(pydoxtools.operators_base.Operator):
    def __call__(self, pandoc_document: "pandoc.types.Pandoc", output_format: str) -> str:
        full_text = pandoc.write(pandoc_document, format=output_format)
        return full_text


class PandocToPdxConverter(pydoxtools.operators_base.Operator):
    """convert pandoc elemens in our "own" element format"""

    def __call__(self, pandoc_document: "pandoc.types.Pandoc") -> pd.DataFrame:
        # extract subsections
        pdoc = pandoc_document
        metadata = pdoc[0]
        pdx_elements = []  # declare our own element "format"
        section_title = None
        boxnum = 0
        for el in pdoc[1]:
            if isinstance(el, pandoc.types.Header):
                section_title = PandocConverter()(el, output_format="plain").strip()
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.Text,
                    sections=[section_title],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=section_title,
                    level=0,
                    boxnum=boxnum
                )
            else:
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.Text,
                    sections=[section_title],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=PandocConverter()(el, output_format="plain").strip(),
                    level=1,
                    boxnum=boxnum
                )
            pdx_elements.append(pdx_el)
        df = pd.DataFrame(pdx_elements)
        return df


class PandocOperator(pydoxtools.operators_base.Operator):
    """
    Extract tables, headers and lists from a pandoc document
    """

    # TODO: also find lists, that are nested inside tables for example...
    #       this means we would have to iterate over all textblocks and reorganize its structure.
    #       this could be done for example by searching for BulletLists, that are not preceded
    #       by BulletList elements and thus finding the "beginning" of a BulletList we can
    #       now combine consecutive elements into a single list block. Otherwise
    #       we would find every sub-list as a separate entitiy. ...
    # pandoc.types.CodeBlock
    def __init__(self, method: str):
        super().__init__()
        self._method = method

    def __call__(self, pandoc_blocks: list["pandoc.types.Block"]) -> str | list[str] | list[pd.DataFrame]:
        if self._method == "headers":
            headers = [pandoc.write(elt[2], format="markdown").strip() for elt in pandoc_blocks
                       if isinstance(elt, pandoc.types.Header)]
            return headers
        elif self._method == "tables_df":
            # txtblocks = [elt for elt in pandoc.iter(doc) if isinstance(elt, pandoc.types.Block)]
            tables = [pd.read_html(pandoc.write(elt, format="html"))[0] for elt in pandoc_blocks if
                      isinstance(elt, pandoc.types.Table)]
            return tables
        elif self._method == "lists":
            olists = [extract_list(elt) for elt in pandoc_blocks if isinstance(elt, (pandoc.types.OrderedList,))]
            blists = [extract_list(elt) for elt in pandoc_blocks if isinstance(elt, (pandoc.types.BulletList,))]
            return olists + blists
