import pandas as pd
import pandoc
import pandoc.types
import logging
from packaging import version
from pydoxtools import document_base

logger = logging.getLogger(__name__)

pandoc_version = pandoc._configuration['version']
if version.parse(pandoc_version) < version.parse('2.14.2'):
    logger.warning(f"installed pandoc version {pandoc_version}, which doesn't support rtf file format!"
                   f"in order to be able to use rtf, you need to install a pandoc version >= 2.14.2")


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


class PandocLoader(document_base.Extractor):
    """
    Converts a string or a raw byte string into pandoc intermediate format.
    """

    def __init__(self):
        super().__init__()

    def __call__(
            self, raw_content: bytes | str, document_type: str
    ) -> pandoc.types.Pandoc:
        pandoc_format = pandoc.read(raw_content, format=document_type.strip("."))
        return pandoc_format


class PandocBlocks(document_base.Extractor):
    def __init__(self):
        super().__init__()

    def __call__(self, pandoc_document: pandoc.types.Pandoc) -> list[pandoc.types.Block]:
        txtblocks = [elt for elt in pandoc_document[1] if isinstance(elt, pandoc.types.Block)]
        return txtblocks


class PandocConverter(document_base.Extractor):
    def __init__(self):
        super().__init__()

    def __call__(self, pandoc_document: pandoc.types.Pandoc, output_format: str = "markdown") -> str:
        full_text = pandoc.write(pandoc_document, format=output_format)
        return full_text


class PandocExtractor(document_base.Extractor):
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
    def __init__(self, method: str, output_format="markdown"):
        super().__init__()
        self._method = method
        self._output_format = output_format

    def __call__(self, pandoc_blocks: list[pandoc.types.Block]) -> str | list[str] | list[pd.DataFrame]:
        if self._method == "headers":
            headers = [pandoc.write(elt[2], format=self._output_format).strip() for elt in pandoc_blocks
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
