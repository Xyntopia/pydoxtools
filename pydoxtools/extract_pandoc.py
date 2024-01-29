from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import mimetypes

import pandas as pd
from packaging import version

import pydoxtools.operators_base

logger = logging.getLogger(__name__)

import pandoc

pydoxtools_link = "https://github.com/jgm/pandoc/releases"

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
    
    Checkout this link here: {pydoxtools_link}. In order to work correctly, pandoc version 3.1.X is needed...
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


class PandocConverter(pydoxtools.operators_base.Operator):
    def __call__(self, pandoc_document: "pandoc.types.Pandoc", output_format: str) -> str:
        full_text = pandoc.write(pandoc_document, format=output_format)
        return full_text


class PandocToPdxConverter(pydoxtools.operators_base.Operator):
    """convert pandoc elemens in our "own" element format"""

    # TODO: also find lists, that are nested inside tables for example...
    #       this means we would have to iterate over all textblocks and reorganize its structure.
    #       this could be done for example by searching for BulletLists, that are not preceded
    #       by BulletList elements and thus finding the "beginning" of a BulletList we can
    #       now combine consecutive elements into a single list block. Otherwise
    #       we would find every sub-list as a separate entitiy. ...

    def __init__(self):
        super().__init__()

    def __call__(self, pandoc_document: "pandoc.types.Pandoc") -> list[pydoxtools.document_base.DocumentElement]:
        # extract subsections
        pdoc = pandoc_document
        pdx_elements = []  # declare our own element "format"
        section_title = None
        boxnum = 0
        for id, el in enumerate(pdoc[1]):  # [0] is potential metadata
            if isinstance(el, pandoc.types.Header):
                section_title = PandocConverter()(el, output_format="plain").strip()
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.Header,
                    sections=[section_title],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=section_title,
                    level=0,
                    boxnum=boxnum,
                    obj=el
                )
            elif isinstance(el, pandoc.types.Table):
                table = pd.read_html(pandoc.write(el, format="html"))[0].to_dict('index')
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.Table,
                    sections=[section_title] if section_title else [],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=PandocConverter()(el, output_format="plain").strip(),
                    level=1,
                    boxnum=boxnum,
                    obj=table
                )
            elif isinstance(el, (pandoc.types.OrderedList, pandoc.types.BulletList)):
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.List,
                    sections=[section_title] if section_title else [],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=PandocConverter()(el, output_format="plain").strip(),
                    level=1,
                    boxnum=boxnum,
                    obj=el
                )
            else:
                boxnum += 1
                pdx_el = pydoxtools.document_base.DocumentElement(
                    type=pydoxtools.document_base.ElementType.Text,
                    sections=[section_title] if section_title else [],
                    rawtext=PandocConverter()(el, output_format="markdown").strip(),
                    text=PandocConverter()(el, output_format="plain").strip(),
                    level=1,
                    boxnum=boxnum,
                    obj=el
                )

            pdx_elements.append(pdx_el)

        return pdx_elements
