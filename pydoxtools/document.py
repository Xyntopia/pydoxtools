import langdetect
import numpy as np
import pandas as pd

from . import document_base
from .extract_classes import LanguageExtractor, TextBlockClassifier
from .extract_files import FileLoader
from .extract_html import HtmlExtractor
from .extract_index import IndexExtractor, KnnQuery, SimilarityGraph, ExtractKeywords
from .extract_logic import Alias, Constant
from .extract_logic import LambdaExtractor
from .extract_nlpchat import OpenAIChat
from .extract_objects import EntityExtractor
from .extract_ocr import OCRExtractor
from .extract_pandoc import PandocLoader, PandocExtractor, PandocConverter, PandocBlocks
from .extract_spacy import SpacyExtractor, extract_spacy_token_vecs, get_spacy_embeddings, extract_noun_chunks
from .extract_tables import ListExtractor, TableCandidateAreasExtractor
from .extract_textstructure import DocumentElementFilter, TextBoxElementExtractor, TitleExtractor
from .html_utils import get_text_only_blocks
from .list_utils import flatten
from .pdf_utils import PDFFileLoader
from .qamachine import QamExtractor
from .settings import settings


class Document(document_base.Pipeline):
    """Standard document class for document analysis, data extraction and transformation.

This class implements an extensive pipeline using the [][document_base.Pipeline] class
for information extraction from documents.

***

In order to load a document, simply
open it with the document class::

    from pydoxtools import Document
    doc = Document(fobj=./data/demo.docx)

You can then access any extracted data by
calling x with the specified member::

    doc.x("addresses")
    doc.x("entities")
    doc.x("full_text")
    # etc...

Most members are also callable like a n
ormal class member in roder to make the code easier to read::

    doc.addresses

A list of all available extraction data
can be called like this::

    doc.x_funcs()

***

The document class is backed by a *pipeline* class with a pre-defined pipeline focusing on
document extraction tasks. This extraction pipeline can be overwritten partially or completly replaced.
In order to customize the pipeline it is usually best to take the pipeline for
basic documents defined in pydoxtools.Document as a starting point and
only overwrite the parts that should be customized.

inherited classes can override any part of the graph.

It is possible to exchange/override/extend or introduce extraction pipelines for individual file types (including
the generic one: "*") such as *.html extractors, *.pdf, *.txt etc..

Strings inside a document class indicate the inclusion of that document type pipeline but with a lower priority
this way a directed extraction graph gets built. This only counts for the current class that is
being defined though!!

Example extension pipeline for an OCR extractor which converts images into text
"image" code block and supports filetypes: ".png", ".jpeg", ".jpg", ".tif", ".tiff"::

    "image": [
            OCRExtractor()
            .pipe(file="raw_content")
            .out("ocr_pdf_file")
            .cache(),
        ],
    # the first base doc types have priority over the last ones
    # so here .png > image > .pdf
    ".png": ["image", ".pdf"],
    ".jpeg": ["image", ".pdf"],
    ".jpg": ["image", ".pdf"],
    ".tif": ["image", ".pdf"],
    ".tiff": ["image", ".pdf"],
    # the "*" gets overwritten by functions above
    "*": [...]

Each function (or node) in the extraction pipeline gets fed its input-parameters
by the "pipe" command. These parameters can be configured on document creation
if some of them are declared using the "config" command.

These arguments can be overwritten by a new pipeline in inherited documents or document types
that are higher up in the hierarchy. The argument precedence is hereby as follows::

    python-class-member < extractor-graph-function < config
"""

    """
    TODO: One can also change the configuration of individual extractors. For example
    of the Table Extractor or Space models...

    TODO: add "extension/override" logic for individual file types. The main important thing there is
          to make sure we don't have any "dangling" functions left over when filetype logics
          gets overwritten
    """

    _extractors = {
        ".pdf": [
            FileLoader()  # pdfs are usually in binary format...
            .pipe(fobj="_fobj").out("raw_content").cache(),
            PDFFileLoader()
            .pipe(fobj="raw_content", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("pages_bbox", "elements", "meta", pages="page_set")
            .cache(),
            LambdaExtractor(lambda pages: len(pages))
            .pipe(pages="page_set").out("num_pages").cache(),
            DocumentElementFilter(element_type=document_base.ElementType.Line)
            .pipe("elements").out("line_elements").cache(),
            DocumentElementFilter(element_type=document_base.ElementType.Graphic)
            .pipe("elements").out("graphic_elements").cache(),
            ListExtractor().cache()
            .pipe("line_elements").out("lists"),
            TableCandidateAreasExtractor()
            .pipe("graphic_elements", "line_elements", "pages_bbox", "text_box_elements", "filename")
            .out("table_candidates", box_levels="table_box_levels").cache(),
            LambdaExtractor(lambda candidates: [t.df for t in candidates if t.is_valid])
            .pipe(candidates="table_candidates").out("table_df0").cache(),
            LambdaExtractor(lambda table_df0, lists: table_df0 + [lists]).cache()
            .pipe("table_df0", "lists").out("tables_df"),
            TextBoxElementExtractor()
            .pipe("line_elements").out("text_box_elements").cache(),
            LambdaExtractor(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").out("text_box_list").cache(),
            LambdaExtractor(lambda tb: "\n\n".join(tb))
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
            LambdaExtractor(lambda article: article.links)
            .pipe(article="goose_article").out("urls").cache(),
            LambdaExtractor(lambda article: article.top_image)
            .pipe(article="goose_article").out("main_image").cache(),
            Alias(full_text="main_content"),
            LambdaExtractor(lambda x: pd.DataFrame(get_text_only_blocks(x), columns=["text"])).cache()
            .pipe(x="raw_content").out("text_box_elements"),
            LambdaExtractor(lambda t, s: [t, s])
            .pipe(t="title", s="short_title").out("titles").cache(),
            LambdaExtractor(lambda x: set(w.strip() for w in x.split(",")))
            .pipe(x="html_keywords_str").out("html_keywords"),  # todo add a generic keywords extraction here
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
            PandocConverter()
            .pipe(pandoc_document="pandoc_document").config(output_format="markdown")
            .out("full_text").cache(),
            PandocBlocks()
            .pipe(pandoc_document="pandoc_document").out("pandoc_blocks").cache(),
            PandocExtractor(method="headers", output_format="markdown")
            .pipe(pandoc_blocks="pandoc_blocks").out("headers").cache(),
            PandocExtractor(method="tables_df", output_format="markdown")
            .pipe(pandoc_blocks="pandoc_blocks").out("tables_df").cache(),
            PandocExtractor(method="lists", output_format="markdown")
            .pipe(pandoc_blocks="pandoc_blocks").out("lists").cache(),
        ],
        "image": [
            # add a "base-document" type (.pdf) images get converted into pdfs
            # and then further processed from there
            ".pdf",  # as we are extracting a pdf we would like to use the pdf functions...
            OCRExtractor()
            .pipe(file="raw_content")
            .out("ocr_pdf_file")
            .config("ocr_lang", ocr="ocr_on")  # TODO move this into BilledDocument and make sure
            .cache(),
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
        "*": [
            FileLoader()
            .pipe(fobj="_fobj", document_type="document_type", page_numbers="_page_numbers", max_pages="_max_pages")
            .out("raw_content").cache(),
            Alias(full_text="raw_content"),
            LambdaExtractor(lambda x: pd.DataFrame(x.split("\n"), columns=["text"]))
            .pipe(x="full_text").out("text_box_elements").cache(),
            LambdaExtractor(lambda df: df.get("text", None).to_list())
            .pipe(df="text_box_elements").out("text_box_list").cache(),
            LambdaExtractor(lambda tables_df: [df.to_dict('index') for df in tables_df]).cache()
            .pipe("tables_df").out("tables_dict"),
            Alias(tables="tables_dict"),
            TextBlockClassifier(min_prob=0.6)
            .pipe("text_box_elements").out("addresses").cache(),
            LambdaExtractor(lambda full_text: 1 + (len(full_text) // 1000))
            .pipe("full_text").out("num_pages").cache(),
            LambdaExtractor(lambda full_text: langdetect.detect(full_text))
            .pipe("full_text").out("language").cache(),

            #########  SPACY WRAPPERS  #############
            SpacyExtractor(model_size="md")
            .pipe("full_text", "language").out(doc="spacy_doc", nlp="spacy_nlp").cache(),
            LambdaExtractor(extract_spacy_token_vecs)
            .pipe("spacy_doc").out("spacy_vectors"),
            LambdaExtractor(get_spacy_embeddings)
            .pipe("spacy_nlp").out("spacy_embeddings"),
            LambdaExtractor(lambda spacy_doc: list(spacy_doc.sents))
            .pipe("spacy_doc").out("spacy_sents"),
            LambdaExtractor(extract_noun_chunks)
            .pipe("spacy_doc").out("spacy_noun_chunks").cache(),
            ########## END OF SPACY ################

            EntityExtractor().cache()
            .pipe("spacy_doc").out("entities"),
            # TODO: try to implement as much as possible from the constants below for all documentypes
            # TODO: implement an automatic summarizer based on textrank...
            Constant(summary="unknown", urls=[], main_image=None, html_keywords=[],
                     final_urls=[], pdf_links=[], schemadata={}, tables_df=[]),
            Alias(url="source"),

            ########### NOUN_INDEX #############
            Alias(noun_chunks="spacy_noun_chunks"),
            LambdaExtractor(
                lambda x: dict(
                    noun_vecs=np.array([e.vector for e in x]),
                    noun_ids=list(range(len(x))))
            )
            .pipe(x="noun_chunks").out("noun_vecs", "noun_ids").cache(),
            IndexExtractor()
            .pipe(vecs="noun_vecs", ids="noun_ids").out("noun_index").cache(),
            LambdaExtractor(lambda spacy_nlp: lambda x: spacy_nlp(x).vector)
            .pipe("spacy_nlp").out("vectorizer").cache(),
            KnnQuery().pipe(index="noun_index", idx_values="noun_chunks", vectorizer="vectorizer")
            .out("noun_query").cache(),
            SimilarityGraph().pipe(index_query_func="noun_query", source="noun_chunks")
            .out("noun_graph").cache(),
            ExtractKeywords(top_k=5).pipe(G="noun_graph").out("textrank_keywords").cache(),
            ########### END NOUN_INDEX ###########

            ########### AGGREGATION ##############
            LambdaExtractor(lambda **kwargs: set(flatten(kwargs.values())))
            .pipe("html_keywords", "textrank_keywords").out("keywords").cache(),

            ########### QaM machine #############
            # TODO: make sure we can set the model that we want to use dynamically!
            QamExtractor(model_id=settings.PDXT_STANDARD_QAM_MODEL)
            .pipe(text="full_text").out("answers").cache().config(trf_model_id="qam_model_id"),

            ########### Chat AI ##################
            OpenAIChat()
            .pipe(full_text="full_text").out("chat_answers").cache().config(model_id="model_id"),
        ]
    }
