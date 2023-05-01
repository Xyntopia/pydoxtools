import mimetypes
from functools import cached_property
from pathlib import Path
from typing import IO, Callable
from urllib.parse import urlparse

import langdetect
import numpy as np
import pandas as pd
import requests
import yaml

import pydoxtools.operators
from . import document_base
from .document_base import Pipeline, ElementType
from .extract_classes import LanguageExtractor, TextBlockClassifier
from .extract_filesystem import FileLoader, PathLoader
from .extract_html import HtmlExtractor
from .extract_index import IndexExtractor, KnnQuery, \
    SimilarityGraph, TextrankOperator, TextPieceSplitter
from .extract_nlpchat import OpenAIChat
from .extract_objects import EntityExtractor
from .extract_ocr import OCRExtractor
from .extract_pandoc import PandocLoader, PandocOperator, PandocConverter, PandocBlocks
from .extract_spacy import SpacyOperator, extract_spacy_token_vecs, get_spacy_embeddings, extract_noun_chunks
from .extract_tables import ListExtractor, TableCandidateAreasExtractor, Iterator2Dataframe
from .extract_textstructure import DocumentElementFilter, TextBoxElementExtractor, TitleExtractor
from .html_utils import get_text_only_blocks
from .list_utils import flatten, flatten_dict, deep_str_convert, iterablefyer
from .nlp_utils import calculate_string_embeddings, summarize_long_text
from .operator_huggingface import QamExtractor
from .operators import Alias, LambdaOperator, ElementWiseOperator, Configuration, Constant, DataMerger
from .pdf_utils import PDFFileLoader


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class DocumentTypeError(Exception):
    pass


class Document(Pipeline):
    """Basic document pipeline class to analyze documents from all kinds of formats.

The Document class is designed for information extraction from documents. It inherits from the
[pydoxtools.document_base.Pipeline][] class and uses a predefined extraction pipeline f
ocused on document processing tasks.
To load a document, create an instance of the Document class with a file path, a file object, a string,
a URL or give it some data directly as a dict:

```
from pydoxtools import Document
doc = Document(fobj=Path('./data/demo.docx'))
```

Extracted data can be accessed by calling the `x` method with the specified
output in the pipeline:

```python
doc.x("addresses")
doc.x("entities")
doc.x("full_text")
# etc...
```

Most members can also be called as normal class attributes for easier readability:

```python
doc.addresses
```

Additionally, it is possible to get the data directly in dict, yaml or json form:

```python
doc.property_dict("addresses","filename","keywords")
doc.yaml("addresses","filename","keywords")
doc.json("addresses","filename","keywords")
```

To retrieve a list of all available extraction data methods, call the `x_funcs()` method:

```python
doc.x_funcs()
```

## Customizing the Pipeline:

The extraction pipeline can be partially overwritten or completely replaced to customize the
document processing. To customize the pipeline, it's recommended to use the basic document
pipeline defined in `pydoxtools.Document` as a starting point and only overwrite parts as needed.

Inherited classes can override any part of the graph. To exchange, override, extend or introduce
extraction pipelines for specific file types (including the generic one: "*"), such as *.html,
*.pdf, *.txt, etc., follow the example below.

TODO: provide more information on how to customize the pipeline and override the graph.

## Examples

The following is an example extension pipeline for an OCR extractor that converts images into
text and supports file types: ".png", ".jpeg", ".jpg", ".tif", ".tiff":

```python
"image": [
        OCRExtractor()
        .pipe(file="raw_content")
        .out("ocr_pdf_file")
        .cache(),
    ],
".png": ["image", ".pdf"],
".jpeg": ["image", ".pdf"],
".jpg": ["image", ".pdf"],
".tif": ["image", ".pdf"],
".tiff": ["image", ".pdf"],
"*": [...]
```

Each function (or node) in the extraction pipeline connects to other nodes in the pipeline
through the "pipe" command. Arguments can be overwritten by a new pipeline in inherited
documents or document types higher up in the hierarchy. The argument precedence is as follows:

```
python-class-member < extractor-graph-function < configuration
```

When creating a new pipeline for documentation purposes, use a function or class for complex
operations and include the documentation there. Lambda functions should not be used in this case.
"""

    """
    TODO: One can also change the configuration of individual operators. For example
    of the Table Operator or Space models...

    TODO: add "extension/override" logic for individual file types. The main important thing there is
          to make sure we don't have any "dangling" functions left over when filetype logics
          gets overwritten
    """

    # TODO: rename extractors to operators
    _operators = {
        ".pdf": [
            FileLoader()  # pdfs are usually in binary format...
            .pipe(fobj="_fobj").out("raw_content").cache(),
            PDFFileLoader()
            .pipe(fobj="raw_content", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("pages_bbox", "elements", "meta", pages="page_set")
            .cache(),
            LambdaOperator(lambda pages: len(pages))
            .pipe(pages="page_set").out("num_pages").cache(),
            DocumentElementFilter(element_type=ElementType.Line)
            .pipe("elements").out("line_elements").cache(),
            DocumentElementFilter(element_type=ElementType.Graphic)
            .pipe("elements").out("graphic_elements").cache(),
            ListExtractor().cache()
            .pipe("line_elements").out("lists"),
            TableCandidateAreasExtractor()
            .pipe("graphic_elements", "line_elements", "pages_bbox", "text_box_elements", "filename")
            .out("table_candidates", box_levels="table_box_levels").cache(),
            LambdaOperator(lambda candidates: [t.df for t in candidates if t.is_valid])
            .pipe(candidates="table_candidates").out("table_df0").cache(),
            LambdaOperator(lambda table_df0, lists: table_df0 + [lists]).cache()
            .pipe("table_df0", "lists").out("tables_df"),
            TextBoxElementExtractor()
            .pipe("line_elements").out("text_box_elements").cache(),
            LambdaOperator(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").out("text_box_list").cache(),
            LambdaOperator(lambda tb: "\n\n".join(tb))
            .pipe(tb="text_box_list").out("full_text").cache(),
            TitleExtractor()
            .pipe("line_elements").out("titles", "side_titles").cache(),
            LanguageExtractor().cache()
            .pipe(text="full_text").out("language").cache()
        ],
        ".html": [
            HtmlExtractor()
            .pipe(raw_html="raw_content", url="source")
            .out("main_content_clean_html", "summary", "language", "goose_article",
                 "main_content", "schemadata", "final_urls", "pdf_links", "title",
                 "short_title", "url", tables="tables_df", html_keywords="html_keywords_str").cache(),
            LambdaOperator(lambda article: article.links)
            .pipe(article="goose_article").out("urls").cache(),
            LambdaOperator(lambda article: article.top_image)
            .pipe(article="goose_article").out("main_image").cache(),
            Alias(full_text="main_content"),
            LambdaOperator(lambda x: pd.DataFrame(get_text_only_blocks(x), columns=["text"])).cache()
            .pipe(x="raw_content").out("text_box_elements"),
            LambdaOperator(lambda t, s: [t, s])
            .pipe(t="title", s="short_title").out("titles").cache(),
            LambdaOperator(lambda x: set(w.strip() for w in x.split(",")))
            .pipe(x="html_keywords_str").out("html_keywords"),

            ########### AGGREGATION ##############
            LambdaOperator(lambda **kwargs: set(flatten(kwargs.values())))
            .pipe("html_keywords", "textrank_keywords").out("keywords").cache(),
        ],
        ".docx": ["pandoc"],
        ".odt": ["pandoc"],
        ".md": ["pandoc"],
        ".rtf": ["pandoc"],
        ".epub": ["pandoc"],
        ".markdown": ["pandoc"],
        "pandoc": [
            PandocLoader()
            .pipe(raw_content="raw_content", document_type="document_type")
            .out("pandoc_document").cache(),
            Configuration(output_format="markdown"),
            PandocConverter()
            .pipe("output_format", pandoc_document="pandoc_document")
            .out("full_text").cache(),
            Constant(clean_format="plain"),
            PandocConverter()  # clean for downstram processing tasks
            .pipe(output_format="clean_format", pandoc_document="pandoc_document")
            .out("clean_text").cache(),
            PandocBlocks()
            .pipe(pandoc_document="pandoc_document").out("pandoc_blocks").cache(),
            PandocOperator(method="headers")
            .pipe(pandoc_blocks="pandoc_blocks").out("headers").cache(),
            PandocOperator(method="tables_df")
            .pipe(pandoc_blocks="pandoc_blocks").out("tables_df").cache(),
            PandocOperator(method="lists")
            .pipe(pandoc_blocks="pandoc_blocks").out("lists").cache()
        ],
        "image": [
            # add a "base-document" type (.pdf) images get converted into pdfs
            # and then further processed from there
            ".pdf",  # as we are extracting a pdf we would like to use the pdf functions...
            Configuration(ocr_lang="auto", ocr_on=True),
            OCRExtractor()
            .pipe("ocr_on", "ocr_lang", file="raw_content")
            .out("ocr_pdf_file"),
            # we need to do overwrite the pdf loading for images we inherited from
            # the ".pdf" logic as we are
            # now taking the pdf from a different variable
            PDFFileLoader()
            .pipe(fobj="ocr_pdf_file")
            .out("pages_bbox", "elements", "meta", pages="page_set")
            .cache(),
        ],
        # the first base doc types have priority over the last ones
        # so here .png > image > .pdf
        ".png": ["image", ".pdf"],
        ".jpeg": ["image", ".pdf"],
        ".jpg": ["image", ".pdf"],
        ".tif": ["image", ".pdf"],
        ".tiff": ["image", ".pdf"],
        ".yaml": [
            "dict",
            Alias(full_text="raw_content"),
            LambdaOperator(lambda x: dict(data=yaml.unsafe_load(x)))
            .pipe(x="full_text").out("data").cache()
            # TODO: we might need to have a special "result" message, that we
            #       pass around....
        ],
        "dict": [  # pipeline to handle data based documents
            Alias(raw_content="_fobj"),
            Alias(data="raw_content"),
            LambdaOperator(lambda x: yaml.dump(deep_str_convert(x)))
            .pipe("data").out("full_text"),
            LambdaOperator(lambda x: [str(k) + ": " + str(v) for k, v in flatten_dict(x.data).items()])
            .pipe(x="data").out("text_box_elements").cache(),
            Alias(text_box_list="text_box_elements"),
            Alias(text_segments="text_box_elements"),
        ],
        # TODO: json, csv etc...
        # TODO: pptx, odp etc...
        "*": [
            # Loading text files
            FileLoader()
            .pipe(fobj="_fobj", document_type="document_type", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("raw_content").cache(),
            Alias(full_text="raw_content"),
            Alias(clean_text="full_text"),

            ## Standard text splitter for splitting text along lines...
            LambdaOperator(lambda x: pd.DataFrame(x.split("\n\n"), columns=["text"]))
            .pipe(x="full_text").out("text_box_elements").cache(),
            LambdaOperator(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").out("text_box_list").cache(),
            # TODO: replace this with a real, generic table detection
            #       e.g. running the text through pandoc or scan for html tables
            Constant(tables_df=[]),
            LambdaOperator(lambda tables_df: [df.to_dict('index') for df in tables_df]).cache()
            .pipe("tables_df").out("tables_dict"),
            Alias(tables="tables_dict"),
            TextBlockClassifier()
            .pipe("text_box_elements").out("addresses").cache(),

            ## calculate some metadata values
            LambdaOperator(lambda full_text: 1 + (len(full_text) // 1000))
            .pipe("full_text").out("num_pages").cache(),
            LambdaOperator(lambda clean_text: len(clean_text.split()))
            .pipe("clean_text").out("num_words").cache(),
            LambdaOperator(lambda spacy_sents: len(spacy_sents))
            .pipe("spacy_sents").out("num_sents"),
            LambdaOperator(lambda ft: sum(1 for c in ft if c.isdigit()) / sum(1 for c in ft if c.isalpha()))
            .pipe(ft="full_text").out("a_d_ratio").cache(),
            LambdaOperator(lambda full_text: langdetect.detect(full_text))
            .pipe("full_text").out("language").cache(),

            #########  SPACY WRAPPERS  #############
            Configuration(spacy_model_size="md", spacy_model="auto"),
            SpacyOperator()
            .pipe(
                "language", "spacy_model",
                full_text="clean_text", model_size="spacy_model_size"
            ).out(doc="spacy_doc", nlp="spacy_nlp").cache(),
            LambdaOperator(extract_spacy_token_vecs)
            .pipe("spacy_doc").out("spacy_vectors"),
            LambdaOperator(get_spacy_embeddings)
            .pipe("spacy_nlp").out("spacy_embeddings"),
            LambdaOperator(lambda spacy_doc: list(spacy_doc.sents))
            .pipe("spacy_doc").out("spacy_sents"),
            LambdaOperator(extract_noun_chunks)
            .pipe("spacy_doc").out("spacy_noun_chunks").cache(),
            ########## END OF SPACY ################

            EntityExtractor().cache()
            .pipe("spacy_doc").out("entities"),
            # TODO: try to implement as much as possible from the constants below for all documentypes
            #       summary, urls, main_image, keywords, final_url, pdf_links, schemadata, tables_df
            # TODO: implement summarizer based on textrank
            Alias(url="source"),

            ########### VECTORIZATION ##########
            Alias(sents="spacy_sents"),
            Alias(noun_chunks="spacy_noun_chunks"),

            LambdaOperator(lambda x: x.vector)
            .pipe(x="spacy_doc").out("vector").cache(),
            LambdaOperator(
                lambda x: dict(
                    sent_vecs=np.array([e.vector for e in x]),
                    sent_ids=list(range(len(x)))))
            .pipe(x="sents").out("sent_vecs", "sent_ids").cache(),
            LambdaOperator(
                lambda x: dict(
                    noun_vecs=np.array([e.vector for e in x]),
                    noun_ids=list(range(len(x)))))
            .pipe(x="noun_chunks").out("noun_vecs", "noun_ids").cache(),

            ########### SEGMENT_INDEX ##########
            TextPieceSplitter()
            .pipe(full_text="full_text").out("text_segments").cache(),
            Configuration(
                text_segment_model="deepset/minilm-uncased-squad2",
                text_segment_only_tokenizer=True
            ),
            ElementWiseOperator(calculate_string_embeddings, return_iterator=False)
            .pipe(
                elements="text_segments",
                model_id="text_segment_model",
                only_tokenizer="text_segment_only_tokenizer")
            .out("text_segment_vectors"),

            ########### NOUN_INDEX #############
            IndexExtractor()
            .pipe(vecs="noun_vecs", ids="noun_ids").out("noun_index").cache(),
            LambdaOperator(lambda spacy_nlp: lambda x: spacy_nlp(x).vector)
            .pipe("spacy_nlp").out("vectorizer").cache(),
            KnnQuery().pipe(index="noun_index", idx_values="noun_chunks", vectorizer="vectorizer")
            .out("noun_query").cache(),
            SimilarityGraph().pipe(index_query_func="noun_query", source="noun_chunks")
            .out("noun_graph").cache(),
            Configuration(top_k_text_rank_keywords=5),
            TextrankOperator()
            .pipe(top_k="top_k_text_rank_keywords", G="noun_graph").out("textrank_keywords").cache(),
            # TODO: we will probably get better keywords if we first get the most important sentences or
            #       a summary and then exract keywords from there :).
            Alias(keywords="textrank_keywords"),
            ########### END NOUN_INDEX ###########

            ########### SENTENCE_INDEX ###########
            IndexExtractor()
            .pipe(vecs="sent_vecs", ids="sent_ids").out("sent_index").cache(),
            LambdaOperator(lambda spacy_nlp: lambda x: spacy_nlp(x).vector)
            .pipe("spacy_nlp").out("vectorizer").cache(),
            KnnQuery().pipe(index="sent_index", idx_values="spacy_sents", vectorizer="vectorizer")
            .out("sent_query").cache(),
            SimilarityGraph().pipe(index_query_func="sent_query", source="spacy_sents")
            .out("sent_graph").cache(),
            Configuration(top_k_text_rank_sentences=5),
            TextrankOperator()
            .pipe(top_k="top_k_text_rank_sentences", G="sent_graph").out("textrank_sents").cache(),

            ########### Huggingface Integration #######
            Configuration(
                summarizer_model="sshleifer/distilbart-cnn-12-6",
                summarizer_token_overlap=50,
                summarizer_max_text_len=200,
            ),
            # Configuration(summarizer_model="sshleifer/distilbart-cnn-12-6"),
            # TODO: discover more "ad-hoc-use-cases" for this
            # HuggingfacePipeline(pipeline="summarization")
            # .pipe("property_dict", trf_model_id="summarizer_model").out("summary_func").cache(),
            # LambdaOperator(lambda x, y: x(y))
            # .pipe(x="summary_func", y="full_text").out("summary").cache(),
            LambdaOperator(lambda x, m, to, ml: summarize_long_text(
                x, m, token_overlap=to, max_len=ml
            )).pipe(
                x="clean_text", m="summarizer_model",
                to="summarizer_token_overlap",
                ml="summarizer_max_text_len"
            ).out("summary"),

            ########### QaM machine #############
            # TODO: make sure we can set the model that we want to use dynamically!
            Configuration(qam_model_id='deepset/minilm-uncased-squad2'),
            QamExtractor()
            .pipe("property_dict", trf_model_id="qam_model_id").out("answers").cache(),

            ########### Chat AI ##################
            Configuration(openai_chat_model_id="gpt-3.5-turbo"),
            OpenAIChat().pipe("property_dict", model_id="openai_chat_model_id")
            .out("chat_answers").cache()
        ]
    }

    def __init__(
            self,
            fobj: str | bytes | Path | IO = None,
            source: str | Path = None,
            page_numbers: list[int] = None,
            max_pages: int = None,
            mime_type: str = None,  # TODO: remove this and document type and replace with "type_hint"
            filename: str = None,
            document_type: str = None,
            # TODO: add "auto" for automatic recognition of the type using python-magic
    ):
        """Initialize a Document instance.

        Args:
            fobj
                The file object to load. Depending on the type of object passed:
                - If a string or bytes object: the object itself is the document.
                - If a string representing a URL: the document will be loaded from the URL.
                - If a pathlib.Path object: load the document from the path.
                - If a file object: load the document from the file object (e.g., bytestream).

            source:
                The source of the extracted data (e.g., URL, 'pdfupload', parent-URL, or a path).

            page_numbers:
                A list of specific pages to extract from the document (e.g., in a PDF).

            max_pages:
                The maximum number of pages to extract to protect resources.

            mime_type:
                The MIME type of the document, if available.

            filename:
                The filename of the document, which can sometimes help in determining its purpose.

            document_type:
                The document type to directly specify the extraction logic to be used.

        """

        super().__init__()

        # TODO: move this code into its own little extractor...
        try:
            if is_url(fobj):
                response = requests.get(fobj)
                fobj = response.content
        except:
            pass

        self._fobj = fobj  # file object
        self._source = source or "unknown"
        self._document_type = document_type
        self._mime_type = mime_type
        self._filename = filename
        self._page_numbers = page_numbers
        self._max_pages = max_pages

    @cached_property
    def filename(self) -> str | None:
        """TODO: move this into document pipeline"""
        if hasattr(self._fobj, "name"):
            return self._fobj.name
        elif isinstance(self._fobj, Path):
            return self._fobj.name
        elif self._filename:
            return self._filename
        else:
            return None

    @cached_property
    def path(self):
        if isinstance(self._fobj, Path):
            return self._fobj
        else:
            return self.source

    @cached_property
    def document_type(self):
        """
        detect doc type based on file-ending
        TODO add a doc-type extractor using for example python-magic
        """
        try:
            if self._document_type:
                return self._document_type
            elif self._mime_type:
                return mimetypes.guess_extension(self._mime_type)
            # get type from path suffix
            elif isinstance(self._fobj, Path):
                if self._fobj.exists():
                    return self._fobj.suffix
                elif hasattr(self._fobj, "name"):
                    return Path(self._fobj.name).suffix
            elif isinstance(self._fobj, str) and (self._document_type is None):
                return "generic"

            # for example if it is a string without a type
            # TODO: detect type with python-magic here...
            raise DocumentTypeError(f"Could not find the document type for {self._fobj[-100:]} ...")
        except:
            try:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj} ...")
            except:
                raise DocumentTypeError(f"Could not detect document type for {self._fobj[-100:]} ...")

    @cached_property
    def pipeline_chooser(self):
        if self.document_type in self._pipelines:
            return self.document_type
        else:
            return "*"

    @property
    def source(self) -> str:
        return self._source

    @property
    def fobj(self):
        return self._fobj

    """
    @property
    def final_url(self) -> list[str]:
        ""sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document.""
        return []

    @property
    def parent(self) -> list[str]:
        ""sources that embed this document in some way (for example as a link)
        (for example a product page which embeds
        a link to this document (e.g. a datasheet)
        ""
        return []
    """


class ForgivingExtractIterator(pydoxtools.operators.Operator):
    """
    Creates a loop to extract properties from a list of
    pydoxtools.Document.

    method = "list":
        Iterator can be configured to output either a list property dictionaries

    method = "single_property"
        output of the iterator will be a single, flat list of said property
    """

    def __init__(self, method="list"):
        super().__init__()
        self._method = method

    def __call__(self, doc_list: list[Document]) -> Callable:
        """Define a safe_extract iterator because we want to stay
        flexible here and not put all this data in our memory...."""

        # TODO: define a more "static" propertylist function which
        #       could be configured on documentset instantiation
        if self._method == "list":
            def safe_extract(properties: list[str] | str) -> list[dict]:
                properties = iterablefyer(properties)
                for doc in doc_list:
                    try:
                        props = doc.property_dict(*properties)
                    except pydoxtools.operators.OperatorException:
                        # we just continue  if an error happened. This is why we are "forgiving"
                        if len(properties) > 1:
                            props = {"Error": "OperatorException"}
                        else:
                            continue

                    if len(properties) == 1:
                        yield props[properties[0]]
                    else:
                        yield props

            return safe_extract


class DocumentSet(document_base.Pipeline):
    """
    This class loads an entire set of documents and processes
    it using a pipeline.

    This class is still experimental. Eventually, the document class
    and this class might get merged into one.
    """

    # make sure we can pass configurations to udnerlying documents!

    # TODO: use our pipelines as an "interface" to LLMs. Describing every function
    #       so that the can choose them as tools!

    # TODO: move more and more things from the "Document" pipeline
    #       into this class here...
    #       the reason is we can sort of create an "inception" class
    #       which can create new documentsets out of itself and
    #       analyze them with th same methods...  that way we can go deeper
    #       & deeper & deeper in the analysis. and synthesize new documents...

    # TODO: give this class multi-processing capabilities
    _operators = {
        # TODO: add "fast, slow, medium" pipelines..  e.g. with keywords extraction as fast
        #       etc...
        # TODO: add a pipeline to add a summarizing description whats in every directory
        "directory": [
            # TODO:  add a string filter which can be used to filte paths & db entries
            #        and is simply a bit more generalized ;)
            PathLoader(mode="files")
            .pipe(directory="_directory", exclude="_exclude")
            .out("file_path_list").cache(),
            PathLoader(mode="dirs")
            .pipe(directory="_directory")
            .out("dir_list").cache(),
            LambdaOperator(lambda pl: [Document(p) for p in pl])
            .pipe(pl="file_path_list").out("doc_list").cache(),
            # it is important here to use no_cache, as we need to re-create the iterator every
            # time we want to use it...
            ForgivingExtractIterator(method="list")
            .pipe("doc_list").out("props_iterator").no_cache(),
            Iterator2Dataframe()
            .pipe(iterator="props_iterator").out("props_df").cache(),
            # LambdaDocumentBuilder(),
            LambdaOperator(
                lambda x: x([
                    "filename",
                    "keywords",
                    "document_type",
                    "url",
                    "path",
                    "num_pages",
                    "num_words",
                    "num_sents",
                    "a_d_ratio",
                    "language"]))
            .pipe(x="props_iterator").out("meta_data_iterator").no_cache(),
            LambdaOperator(lambda x: x)
            .pipe(x="props_iterator").out("file_stats").cache(),
            DataMerger()
            .pipe(root_dir="_directory")
            .out(joint_data="meta_data").cache()
        ],
        "*": []
    }

    def __init__(
            self,
            directory: str | Path,
            exclude: list[str] = None
    ):
        super().__init__()
        self._directory = directory
        self._exclude = exclude or []

    @cached_property
    def pipeline_chooser(self) -> str:
        return "directory"
