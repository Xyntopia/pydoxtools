import functools
import io
import json
import logging
import mimetypes
from functools import cached_property
from pathlib import Path
from typing import IO, Protocol
from urllib.parse import urlparse

import langdetect
import numpy as np
import pandas as pd
import pydantic
import requests
import yaml
from dask.bag import Bag

from . import dask_operators
from . import nlp_utils
from .dask_operators import SQLTableLoader
from .document_base import Pipeline, ElementType, Configuration
from .extract_classes import LanguageExtractor, TextBlockClassifier
from .extract_filesystem import FileLoader
from .extract_filesystem import PathLoader
from .extract_html import HtmlExtractor
from .extract_index import IndexExtractor, KnnQuery, \
    SimilarityGraph, TextrankOperator, TextPieceSplitter
from .extract_nlpchat import OpenAIChat
from .extract_objects import EntityExtractor
from .extract_ocr import OCRExtractor
from .extract_pandoc import PandocLoader, PandocOperator, PandocConverter, PandocBlocks
from .extract_spacy import SpacyOperator, extract_spacy_token_vecs, get_spacy_embeddings, extract_noun_chunks
from .extract_tables import Iterator2Dataframe
from .extract_tables import ListExtractor, TableCandidateAreasExtractor
from .extract_textstructure import DocumentElementFilter, TextBoxElementExtractor, TitleExtractor
from .html_utils import get_text_only_blocks
from .list_utils import flatten, flatten_dict, deep_str_convert
from .nlp_utils import calculate_string_embeddings, summarize_long_text
from .operator_huggingface import QamExtractor
from .operators_base import Alias, LambdaOperator, ElementWiseOperator, Constant
from .operators_base import DataMerger, \
    ForgivingExtractIterator
from .pdf_utils import PDFFileLoader

logger = logging.getLogger(__name__)


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class DocumentTypeError(Exception):
    pass


class DocumentInitializationError(Exception):
    pass


# defining a document interface class
class DocumentType(Protocol):
    # TODO: define our minimum document output
    #       with downstream
    # TODO: generate this class by going through
    #       x-funcs automatically
    # TODO: we need at least define an interface which is cmopatible with
    #       the document class
    raw_content: str
    full_text: str

    def answers(self, questions: list[str]) -> list[str]: ...


class Document(Pipeline):
    """Basic document pipeline class to analyze documents from all kinds of formats.

A list and documentation of all document analysis related functions can be found
[->here<-](../pipelines).

The Document class is designed for information extraction from documents. It inherits from the
[pydoxtools.document_base.Pipeline][] class and uses a predefined extraction pipeline
focused on document processing tasks.
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
        # .pdf
        "application/pdf": [
            FileLoader()  # pdfs are usually in binary format...
            .pipe(fobj="fobj").out("raw_content").cache(),
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
        # .html
        "text/html": [
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
        # docx
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ["pandoc"],
        # odt
        "application/vnd.oasis.opendocument.text": ["pandoc"],
        # markdown
        "text/markdown": ["pandoc"],
        # .rtf
        "text/rtf": ["pandoc"],
        # .epub
        "application/epub+zip": ["pandoc"],
        # pandoc document conversion pipeline
        "pandoc": [
            PandocLoader()
            .pipe(raw_content="raw_content", document_type="document_type")
            .out("pandoc_document").cache(),
            Configuration(full_text_format="markdown"),
            PandocConverter()
            .pipe(output_format="full_text_format", pandoc_document="pandoc_document")
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
        # standard image pipeline
        "image": [
            # add a "base-document" type (.pdf) images get converted into pdfs
            # and then further processed from there
            "application/pdf",  # as we are extracting a pdf we would like to use the pdf functions...
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
        'image/png': ["image", "application/pdf"],
        'image/jpeg': ["image", "application/pdf"],
        'image/tiff': ["image", "application/pdf"],
        "application/x-yaml": [
            "<class 'dict'>",
            Alias(full_text="raw_content"),
            LambdaOperator(lambda x: dict(data=yaml.unsafe_load(x)))
            .pipe(x="full_text").out("data").cache()
            # TODO: we might need to have a special "result" message, that we
            #       pass around....
        ],
        # simple dictionary with arbitrary data from python
        "<class 'dict'>": [  # pipeline to handle data based documents
            Alias(raw_content="fobj"),
            Alias(data="raw_content"),
            LambdaOperator(lambda x: yaml.dump(deep_str_convert(x)))
            .pipe(x="data").out("full_text"),
            LambdaOperator(lambda x: pd.DataFrame([
                str(k) + ": " + str(v) for k, v in flatten_dict(x).items()],
                columns=["text"]
            )).pipe(x="data").out("text_box_elements").cache(),
            Alias(text_segments="text_box_list"),
            # TODO: maybe we should declare a new function for this ;)?
            LambdaOperator(lambda x, s: functools.cache(lambda *y: Document({k: x[k] for k in y}, source=s)))
            .pipe(x="data", s="source").out("data_doc").cache().docs(
                "Create new data document from a subset of the data. The new document will have"
                "the same source as the original one."
            ),
            LambdaOperator(lambda x: x.keys())
            .pipe(x="data").out("keys").no_cache(),
            LambdaOperator(lambda x: x.values())
            .pipe(x="data").out("values").no_cache(),
            LambdaOperator(lambda x: x.values())
            .pipe(x="data").out("items").no_cache()
        ],
        # TODO: json, csv etc...
        # TODO: pptx, odp etc...
        "*": [
            # Loading text files
            FileLoader()
            .pipe(fobj="fobj", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("raw_content").cache(),
            Alias(full_text="raw_content"),
            Alias(clean_text="full_text"),

            # adding a dummy-data operator for downstream-compatibiliy. Not every document has
            # explicitly defined data associated with it.
            Constant(data=dict()),

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

            ########### VECTORIZATION (SPACY) ##########
            Alias(sents="spacy_sents"),
            Alias(noun_chunks="spacy_noun_chunks"),

            LambdaOperator(lambda x: x.vector)
            .pipe(x="spacy_doc").out("vector").cache(),
            # TODO: make this configurable.. either we want
            #       to use spacy for this or we would rather have a huggingface
            #       model doing this...
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

            ########### VECTORIZATION (Huggingface) ##########
            Alias(sents="spacy_sents"),
            Alias(noun_chunks="spacy_noun_chunks"),
            Configuration(
                vectorizer_model="deepset/minilm-uncased-squad2",
                vectorizer_only_tokenizer=False
            ).docs("Choose the embeddings model (huggingface-style) and if we want"
                   "to do the vectorization using only the tokenizer. Using only the"
                   "tokenizer is MUCH faster and uses lower CPU than creating actual"
                   "contextual embeddings using the model. BUt is also lower quality"
                   "because it lacks the context."),
            LambdaOperator(
                lambda x, m, t: nlp_utils.calculate_string_embeddings(
                    text=x, model_id=m, only_tokenizer=t)
            ).pipe(x="full_text", m="vectorizer_model", t="vectorizer_only_tokenizer")
            .out("tok_embeddings").cache(),

            ########### SEGMENT_INDEX ##########
            TextPieceSplitter()
            .pipe(full_text="full_text").out("text_segments").cache(),
            ElementWiseOperator(calculate_string_embeddings, return_iterator=False)
            .pipe(
                elements="text_segments",
                model_id="vectorizer_model",
                only_tokenizer="vectorizer_only_tokenizer")
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
            ).out("summary").cache(),

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
            fobj: str | bytes | Path | IO | dict | list | set = None,
            # TODO: only use Path and declare the variables using pydantic we'll also get validation
            source: str | Path = None,
            document_type: str = "auto",
            page_numbers: list[int] = None,
            max_pages: int = None,
    ):
        """Initialize a Document instance.

        Either fobj or source are required. They can both be given. If either of them
        isn't specified the other one is inferred automatically.

        document_type, page_number and max_pages are also not required, but can be used to override
        the default behaviour. specifically document_type can be used manually specify
        the pipeline that should be used.

        Args:
            fobj:
                The file object or data to load. Depending on the type of object passed:
                - If a string or bytes object: the object itself is the document.
                - If a string representing a URL: the document will be loaded from the URL.
                - If a pathlib.Path object: load the document from the path.
                - If a file object: load the document from the file object (e.g., bytestream).
                - If a python dict object: interprete a "dict" as a document
                - If a python list object: interprete a "list" as a document

            source:
                The source of the extracted data (e.g., URL, 'pdfupload', parent-URL, or a path).
                source is given in addition to fobj it overrides the automatically inferred source.
                A special case applies if our document is a dataobject from a database. In that case
                the index key from the database should be used as source. This facilitates downstream
                tasks immensely where we have to refer back to where the data came from.

                This also applies for "explode" operations on documents where the newly created documents
                will all try to trace their origin using the "source" attribute

            document_type:
                The document type to directly specify the pipeline to be used. If "auto" is given
                it will try to be inferred automatically. For example in some cases
                we would like to have a string given in fobj not to be loaded as a file
                but actually be used as raw "string" data. In this case we can explicitly
                specify document_type="string"

            page_numbers:
                A list of specific pages to extract from the document (e.g., in a PDF).

            max_pages:
                The maximum number of pages to extract to protect resources.

        """

        super().__init__()

        # TODO: move this code into its own little extractor...
        #       can also be made better with pydantic ;).
        try:
            if is_url(fobj):
                response = requests.get(fobj)
                fobj = response.content
        except:
            pass

        if not (source or fobj):
            raise DocumentInitializationError(
                "Either 'source' or 'fobj' are required for initialzation of Document Class")

        self._fobj = fobj  # file or data object
        self._source = source
        self._document_type = document_type  # override pipeline selection
        self._page_numbers = page_numbers
        self._max_pages = max_pages

    @cached_property
    def fobj(self) -> bytes | str | Path | IO | dict | list | set:
        if self._fobj:
            return self._fobj
        else:
            return self._source

    @cached_property
    def source(self) -> str:
        return self._source

    @property
    def filename(self) -> str | None:
        """return filename or some other identifier of a file"""
        _, filepath = self.document_type_detection()
        if filepath:
            return filepath.name
        else:
            return "<unknown>"

    @property
    def path(self):
        _, filepath = self.document_type_detection()
        return filepath

    @property
    def document_type(self):
        # document type was overriden
        if self._document_type != "auto":
            return self._document_type
        else:
            doc_type, _ = self.document_type_detection()
            return doc_type

    @functools.cache
    def document_type_detection(self):
        """
        This one here is actually important as it detects the
        type of data that we are going to use for out pipeline.
        That is also why this is implemented as a member function and can
        not be pushed in the pipeline itself, because in needs to be run
        in order to select which pipline we are going to use.

        detect doc type based on various criteria
        TODO add a doc-type extractor using for example python-magic
        """
        buffer = None
        detected_filepath: Path | None = None
        mimetype = "text/plain"  # use this as a standard mimetype

        try:
            import magic
        except ImportError:
            logger.warning("libmagic was not found on your system! falling back"
                           "to filetype detection by file-ending only. In order"
                           "to have a more robust file type detection it would be"
                           "advisable to install this library.")
            magic = False

        # if it works... now, check filetype using python-magic
        if isinstance(self.fobj, (Path, str)):
            fobj = Path(self.fobj)
            if fobj.is_file():  # check if we have an actual file here
                if magic:
                    mimetype = magic.from_file(self.fobj, mime=True)
                detected_filepath = fobj
            else:
                # if it's not a file, we take it as a string, but can be overwritten
                # using the document_type variable during initialization
                mimetype = "string"
        elif isinstance(self.fobj, bytes):
            if magic:
                mimetype = magic.from_buffer(self.fobj, mime=True)
            else:
                logger.warning(f"no filetype specified for {self.source}")
        elif isinstance(self.fobj, io.IOBase):  # might be a streaming object
            # TODO: there is probably a better way to check for many file io objects...
            if magic:
                buffer = self.fobj.read(2048)
                self.fobj.seek(0)  # reset pointer for downstream tasks
                mimetype = magic.from_buffer(buffer, mime=True)

            try:
                detected_filepath = Path(self.fobj.name)
            except:
                pass
        elif isinstance(self.fobj, (dict, list, set)):
            mimetype = str(type(self.fobj))
        else:
            mimetype = str(type(self.fobj))

        # do some more checks on problems that we have encountered when using "magic"
        if mimetype == "application/json":
            if buffer:
                jsonstr = self.fobj.read()
                self.fobj.seek(0)
            elif detected_filepath:
                with open(self.fobj, "r") as f:
                    # TODO: can we make this "lazy" basically throw anexception later, so that the document_type
                    #       can change iterativly?
                    jsonstr = f.read()
            else:
                jsonstr = self.fobj

            try:
                json.loads(jsonstr)
            except json.decoder.JSONDecodeError:
                mimetype = "text/plain"

        # if file is text based or json etc... do some more checks!
        # the file ending becomes a lot more important in this case!
        if mimetype == "text/plain":
            # it's hard to check for the actual filetype here with python magic, so we
            # fall back to using the extension itself
            if detected_filepath or self.source:
                mimetype, encoding = mimetypes.guess_type(detected_filepath or self.source, strict=False)
            return mimetype, detected_filepath
        else:
            return mimetype, detected_filepath

    # TODO: implement a logic on what to do if
    #       mimetypes and magic don't agree'

    @cached_property
    def pipeline_chooser(self):
        if self.document_type in self._pipelines:
            return self.document_type
        else:
            return "*"

    def __repr__(self):
        """
        Returns:
            str: A string representation of the instance.
        """
        if isinstance(self._source, str | bytes):
            return f"{self.__module__}.{self.__class__.__name__}(source={self._source[:10] or self.filename})"
        else:
            return f"{self.__module__}.{self.__class__.__name__}(source={self._source or self.filename})"

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


class DatabaseSource(pydantic.BaseModel):
    connection_string: str
    sql: str
    index_column: str


class DocumentBag(Pipeline):
    """
    **This class is WIP use with caution**

    This class loads an entire set of documents and processes
    it using a pipeline.

    In order to fit into memory we are making extensive use of dask bags to
    store and make calculations on documents.

    For more information check the documentation for dask bags
    [->here<-](https://docs.dask.org/en/stable/bag.html).

    This has the added benefit that one can use dask dataframes out of the box
    for downstream calculations!!

    This class is still experimental. Expect more documentation in
    the near future.

    This class is developed to do work on larger-than-memory datasets and
    scale LLM & AI inference to very large workloads.

    This class makes mostly use of iterative dask bags & dataframes
    to avoid out-of-memory problems.
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

    # TODO: resolve naming conflicts:
    #       bag & documentbag
    _operators = {
        str(DatabaseSource): [
            str(Bag),  # DatabaseSource will eventually be turned into a bag of documents
            LambdaOperator(lambda x: x.dict())
            .pipe(x="source")
            .out("sql", "connection_string", "index_column").cache(),
            SQLTableLoader()
            .pipe("sql", "connection_string", "index_column").out("dataframe").cache(),
            LambdaOperator(lambda x: x.to_bag(index=True, format="dict"))
            .pipe(x="dataframe")
            .out("bag").cache(),
            dask_operators.BagMapOperator(lambda y: Document(y, source=str(y.get('index', None))))
            .pipe(dask_bag="bag").out("docs").cache().docs(
                "Create a dask bag of one data document for each row of the source table"),
        ],
        str(Bag): [
            # TODO: accept arbitrary bags
            # Alias(bag="source"),
            Alias(docs="source"),
            LambdaOperator(lambda x: x.take)
            .pipe(x="docs").out("take").cache(),
            LambdaOperator(lambda x: x.compute)
            .pipe(x="docs").out("compute").cache(),
            # dask_operators.BagMapOperator(lambda x: functools.cache(
            #    lambda y: DocumentBag()))
            # .pipe(dask_bag="dict_bag").out("docbag").cache(),
            dask_operators.BagPropertyExtractor()
            .pipe(dask_bag="docs").out("get_dicts").cache(),
            # get a bag of data documents
            LambdaOperator(lambda docs_bag: lambda *props: docs_bag.map(
                lambda d: d.data_doc(*props)))
            .pipe(docs_bag="docs").out("get_datadocs").cache(),
            LambdaOperator(lambda x: lambda *props: DocumentBag(x(*props)))
            .pipe(x="get_datadocs").out("get_data_docbag").cache(),
        ],
        # TODO: add "fast, slow, medium" pipelines..  e.g. with keywords extraction as fast
        #       etc...
        # TODO: add a pipeline to add a summarizing description whats in every directory
        str(Path): [
            # TODO:  add a string filter which can be used to filte paths & db entries
            #        and is simply a bit more generalized ;)
            # TODO: make this all an iterable...  maybe even using dask..
            Alias(root_path="source"),
            PathLoader(mode="files")
            .pipe(directory="root_path", exclude="_exclude")
            .out("file_path_list").cache(),
            PathLoader(mode="dirs")
            .pipe(directory="root_path")
            .out("dir_list").cache(),
            LambdaOperator(lambda x: [Document(d) for d in x])
            .pipe(data="file_path_list").out("docs").cache(),
            # it is important here to use no_cache, as we need to re-create the iterator every
            # time we want to use it...
            ForgivingExtractIterator(method="list")
            .pipe("docs").out("props_iterator").no_cache(),
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
            .pipe(root_dir="_source")
            .out(joint_data="meta_data").cache()
        ],
        "*": []
    }

    def __init__(
            self,
            source: str | Path | DatabaseSource | Bag,  # can be sql information,
            pipeline: str = None,
            exclude: list[str] = None,
            max_documents: int = None,
    ):
        super().__init__()
        self._source = source
        self._exclude = exclude or []
        self._pipeline = pipeline or "directory"
        self._max_documents = max_documents

    @cached_property
    def source(self):
        return self._source

    @cached_property
    def pipeline_chooser(self) -> str:
        return str(type(self._source))
