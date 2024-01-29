from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import io
import json
import logging
import mimetypes
import re
import tempfile
from functools import cached_property
from pathlib import Path
from typing import IO, Protocol, Any, Callable
from urllib.parse import urlparse

import PIL
import dask.bag
import langdetect
import networkx
import numpy as np
import pandas as pd
import pydantic
import requests
import yaml
from dask.bag import Bag

from . import dask_operators
from . import extract_nlpchat
from . import list_utils
from . import nlp_utils, extract_textstructure
from .dask_operators import SQLTableLoader, DocumentBagMap
from .document_base import Pipeline, ElementType, DocumentElement
from .extract_classes import LanguageExtractor, TextBlockClassifier, PageClassifier
from .extract_filesystem import PathLoader
from .extract_filesystem import force_decode, load_raw_file_content
from .extract_html import HtmlExtractor
from .extract_index import IndexExtractor, KnnQuery, \
    SimilarityGraph, TextrankOperator, TextPieceSplitter
from .extract_objects import EntityExtractor
from .extract_ocr import OCRExtractor
from .extract_pandoc import PandocLoader, PandocConverter, PandocToPdxConverter
from .extract_spacy import (SpacyOperator, extract_spacy_token_vecs, get_spacy_embeddings, extract_noun_chunks,
                            ExtractRelationships, build_document_graph, CoreferenceResolution)
from .extract_tables import ListExtractor, TableCandidateAreasExtractor
from .extract_textstructure import (DocumentElementFilter, text_boxes_from_elements,
                                    TitleExtractor, SectionsExtractor, extract_text_elements)
from .html_utils import get_text_only_blocks
from .list_utils import remove_list_from_lonely_object
from .nlp_utils import calculate_string_embeddings, summarize_long_text
from .operator_huggingface import QamExtractor
from .operators_base import Alias, FunctionOperator, ElementWiseOperator, Constant, DictSelector, \
    Operator, Configuration
from .pdf_utils import PDFFileLoader, PDFImageRenderer

logger = logging.getLogger(__name__)


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def contains_markdown(text: str | bytes) -> bool:
    text = force_decode(text)

    markdown_patterns = [
        r'\*{1,2}[^*]+\*{1,2}',  # Bold or italic: *text* or **text**
        r'#{1,6}\s',  # Headers: # text
        r'\[.*\]\(.*\)',  # Links: [text](url)
        r'!\[.*\]\(.*\)',  # Images: ![text](url)
        r'`[^`]+`',  # Inline code: `code`
        r'^\s{0,3}>\s',  # Blockquotes: > text
        r'^\s{0,3}[-*+]\s',  # Unordered lists: - text or * text or + text
        r'^\s{0,3}\d+\.\s',  # Ordered lists: 1. text
        r'^\s{0,3}(```|~~~)',  # Code blocks: ``` or ~~~
        r'\*\*\*|---|___',  # Horizontal rules: *** or --- or ___
    ]

    match_count = 0
    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            match_count += 1
            if match_count >= 3:
                return True

    return False


def is_html_or_xml(file_content):
    html_pattern = re.compile(r'<!DOCTYPE html.*?>', re.IGNORECASE | re.DOTALL)
    xml_pattern = re.compile(r'<\?xml.*?\?>', re.IGNORECASE | re.DOTALL)

    is_html = bool(html_pattern.search(file_content))
    is_xml = bool(xml_pattern.search(file_content))

    if is_html:
        return 'text/html'
    elif is_xml:
        return 'application/xml'
    else:
        return 'unknown'


def detect_xml_type(file_content):
    mediawiki_pattern = re.compile(r'<mediawiki xmlns="http://www.mediawiki.org/xml/export-.*?"',
                                   re.IGNORECASE | re.DOTALL)
    if bool(mediawiki_pattern.search(file_content)):
        return "mediawiki"
    else:
        return "application/xml"


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


class DocumentBagType(Protocol):
    # TODO: define our minimum document output
    #       with downstream
    # TODO: generate this class by going through
    #       x-funcs automatically
    # TODO: we need at least define an interface which is cmopatible with
    #       the document class
    docs: dask.bag.Bag

    def apply(self, questions: list[str]) -> list[str]: ...


def calculate_a_d_ratio(ft: str) -> float:
    """
    calculate the retaio of digits vs alphabeticsin a string

    can be between 0.0 (no letters) and 1.0 (all letters)
    """
    alphas = sum(1 for c in ft if c.isalpha())
    digits = sum(1 for c in ft if c.isdigit())

    if alphas or digits:
        ratio = alphas / (alphas + digits)
    else:
        ratio = 0.5
    return ratio


# nodes which help to extract the structure of a document
PDFDocumentStructureNodes = [
    extract_textstructure.PDFDocumentObjects()
    .input("valid_tables", "elements", "labeled_text_boxes").out("document_objects").cache(allow_disk_cache=True)
    .docs("extracts a list of document objects such as tables, text boxes, figures, etc."),
    FunctionOperator(lambda tables, elements: {
        k: extract_textstructure.get_bbox_context(v.bbox, elements, v.page, "table")
        for k, v in enumerate(tables)
    }).input("elements", tables="valid_tables").out("table_context").cache()
    .t(dict[int, str])
    .docs("Outputs a dictionary with the context of each table in the document"),
]

PDFNodes = [
    PDFFileLoader()
    .input(fobj="raw_content", page_numbers="_page_numbers", max_pages="_max_pages")
    .out("pages_bbox", "elements", meta="meta_pdf", pages="page_set").cache().docs(
        "Loads a pdf file and returns a list of basic document elements such as lines, figures, etc."),
    Alias(embedded_meta="meta_pdf"),
    Configuration(image_dpi=72 * 3)
    .docs("The dpi when rendering the document."
          " The standard image generation resolution is set to 216 dpi for pdfs"
          " as we want to have sufficient DPI for downstram OCR tasks (e.g."
          " table extraction)"),
    PDFImageRenderer()
    .input(fobj="raw_content", dpi="image_dpi", page_numbers="page_set")
    .out("images").cache()
    .docs("Render a pdf into images which can be used for further downstream processing"),
    FunctionOperator(lambda pages: len(pages)).t(int)
    .input(pages="page_set").out("num_pages").cache()
    .docs("Outputs the number of pages in the document"),

    #########  TABLE STUFF ##############
    ListExtractor().cache()
    .input("line_elements").out("lists")
    .docs("Extracts lists from the document text elements"),
    TableCandidateAreasExtractor()
    .input("graphic_elements", "line_elements", "pages_bbox", "text_box_elements", "filename")
    .out("table_candidates", box_levels="table_box_levels").cache()
    .docs("Detects table candidates from the document elements"),
    FunctionOperator(lambda x: [t for t in x if t.is_valid])
    .input(x="table_candidates").out("valid_tables")
    .docs("Filter valid tables from table candidates by looking if meaningful values can be extracted"),
    FunctionOperator(lambda x: [t.df for t in x])
    .input(x="valid_tables").out("table_df0").cache(allow_disk_cache=True)
    .t(table_df0=list[pd.DataFrame])
    .docs("Filter valid tables from table candidates by looking if meaningful values can be extracted"),
    FunctionOperator(lambda x: pd.DataFrame([t.bbox for t in x], columns=["x0", "y0", "x1", "y1"]))
    .input(x="valid_tables").out("table_areas").cache()
    .t(table_areas=list[np.ndarray])
    .docs("Areas of all detected tables"),
    FunctionOperator(lambda table_df0, lists: table_df0 + ([] if lists.empty else [lists]))
    .cache().input("table_df0", "lists").out("tables_df").cache()
    .t(list[pd.DataFrame])
    .docs("Dataframes of all tables"),
    ############## END TABLE STUFF ##############
    # FunctionOperator()
    # .pipe()
    FunctionOperator(text_boxes_from_elements)
    .input("line_elements").out("text_box_elements").cache()
    .docs("Extracts a dataframe of text boxes from the document by grouping text elements")
    .t(pd.DataFrame),
    FunctionOperator(lambda tb: "\n\n".join(te.text for te in tb)).t(str)
    .input(tb="text_box_elements").out("full_text").cache()
    .docs("Extracts the full text from the document by grouping text elements"),
    TitleExtractor()
    .input("line_elements").out("titles", "side_titles").cache()
    .docs("Extracts the titles from the document by detecting unusual font styles"),
    LanguageExtractor().cache()
    .input(text="full_text").out("language").cache()
    .docs("Extracts the language of the document"),

    *PDFDocumentStructureNodes
]

HTMLNodes = [
    HtmlExtractor()
    .input(raw_html="raw_content", url="source")
    .out("main_content_clean_html", "summary", "language", "goose_article",
         "main_content", "schemadata", "final_urls", "pdf_links", "title",
         "short_title", "url", tables="tables_df", html_keywords="html_keywords_str").cache()
    .docs("Extracts the main content from the html document, removing boilerplate and other noise"),
    FunctionOperator(lambda article: article.links)
    .input(article="goose_article").out("urls").cache()
    .docs("Extracts the urls from the html document"),
    FunctionOperator(lambda article: article.top_image)
    .input(article="goose_article").out("main_image").cache()
    .docs("Extracts the main image from the html document"),
    Alias(full_text="main_content").t(str),
    # TODO: replace this with a new "elements" list.
    FunctionOperator(lambda x: [DocumentElement(
        type=ElementType.TextBox,
        text=tb,
        boxnum=i,
        level=1,
    ) for i, tb in enumerate(get_text_only_blocks(x))]).cache()
    .input(x="raw_content").out("text_box_elements")
    .docs("Extracts the text boxes from the html document"),
    FunctionOperator(lambda t, s: [t, s])
    .input(t="title", s="short_title").out("titles").cache()
    .docs("Extracts the titles from the html document"),
    FunctionOperator(lambda x: {w.strip() for w in x.split(",")})
    .input(x="html_keywords_str").out("html_keywords").cache()
    .docs("Extracts explicitly given keywords from the html document"),

    ########### AGGREGATION ##############
    FunctionOperator(lambda **kwargs: set(list_utils.flatten(kwargs.values())))
    .input("html_keywords", "textrank_keywords").out("keywords").cache()
    .docs("Aggregates the keywords from the html document and found by other algorithms"),
]

# can handle a lot of different input formats
PandocNodes = [
    PandocLoader()
    .input(raw_content="raw_content", document_type="document_type")
    .out("pandoc_document").cache()
    .docs("Loads the document using the pandoc project [https://pandoc.org/](https://pandoc.org/)"
          " into a pydoxtools list of [][pydoxtools.document_base.DocumentElement]"),
    Configuration(full_text_format="markdown")
    .docs("The format used to convert the document to a string"),
    PandocConverter()
    .input(output_format="full_text_format", pandoc_document="pandoc_document")
    .out("full_text").t(str).cache()
    .docs("Converts the document to a string using pandoc"),
    FunctionOperator(lambda x: lambda o: PandocConverter()(x, output_format=o))
    .input(x="pandoc_document").out("convert_to").cache()
    .docs("Generic pandoc converter for other document formats. TODO: better docs"),
    Constant(clean_format="plain")
    .docs("The format used to convert the document to a clean string for downstream processing tasks"),
    FunctionOperator(lambda pdoc: pdoc[0]._arg0)
    .input(pdoc="pandoc_document").out("meta_pandoc")
    .docs("meta information from pandoc document"),
    Alias(embedded_meta="meta_pandoc"),
    PandocToPdxConverter()
    .input("pandoc_document").out("elements").cache().docs(
        "split a pandoc document into text elements."),
    SectionsExtractor()
    .input("elements").out("sections").cache()
    .docs("Extracts the sections from the document by grouping text elements"),
    PandocConverter()  # clean for downstram processing tasks
    .input(output_format="clean_format", pandoc_document="pandoc_document")
    .out("clean_text").cache().docs(
        "for some downstream tasks, it is better to have pure text, without any sructural elements in it")
]

PILImageNodes = [
    "image", "application/pdf",
    FunctionOperator(lambda x: np.array(x))
    .input(x="_fobj").out("data").cache()
    .docs("Converts the image to a numpy array"),
    Alias(pil_image="_fobj"),
]

OCRNodes = [
    # add a "base-document" type (.pdf) images get converted into pdfs
    # and then further processed from there
    "application/pdf",  # as we are extracting a pdf we would like to use the pdf functions...
    Configuration(ocr_lang="auto", ocr_on=True)
    .docs("Configuration for the ocr extractor. We can turn it on/off and specify the language"
          " used for OCR."),
    FunctionOperator(lambda x: PIL.Image.open(io.BytesIO(x)))
    .input(x="raw_content").out("pil_image").cache().t(PIL.Image.Image)
    .docs("Converts the image to a PIL-style image for downstream processing tasks"),
    FunctionOperator(lambda x: np.array(x))
    .input(x="pil_image").out("data").cache().t(np.array)
    .docs("Converts the image to a numpy array for downstream processing tasks"),
    FunctionOperator(lambda x: {0: x}).input(x="pil_image")
    .out("images").no_cache().t(dict[int, PIL.Image.Image])
    .docs("Access images as a dictionary with page numbers as keys for downstream processing tasks "),
    OCRExtractor()
    .input("ocr_on", "ocr_lang", file="raw_content")
    .out("ocr_pdf_file").cache()
    .docs("Extracts the text from the document using OCR. It does this by creating a pdf which "
          "is important in order to keep the positional information of the text elements."),
    # we need to do overwrite the pdf loading for images we inherited from
    # the ".pdf" logic as we are
    # now taking the pdf from a different variable
    PDFFileLoader()
    .input(fobj="ocr_pdf_file")
    .out("pages_bbox", "elements", meta="meta_pdf", pages="page_set")
    .cache().docs("Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]"),
    TableCandidateAreasExtractor(method="images")
    .input("graphic_elements", "line_elements", "pages_bbox", "text_box_elements", "filename",
           "images")
    .out("table_candidates", box_levels="table_box_levels").cache()
    .docs("Extracts the table candidates from the document. As this is an image, we need to "
          "use a different method than for pdfs. Right now this relies on neural networks."
          " TODO: add adtitional pure text-based method."),
]

ClassifierNodes = [
    TextBlockClassifier()
    .t(list[DocumentElement])
    .input("text_box_elements").out("labeled_text_boxes").cache()
    .docs("Classifies the text elements into addresses, emails, phone numbers, etc. if possible."),
    FunctionOperator(lambda x: [t.text for t in x if "address" in t.labels])
    .t(list[str])
    .input(x="labeled_text_boxes").out("addresses")
    .docs("get addresses from text"),
    PageClassifier()
    .input("page_templates").out("page_classifier").cache()
    .docs("Classifies the pages into different types. This is useful for example for "
          "identifiying table of contents, certain chapters etc... . This works "
          "as a zero-shot classifier and the classes are not predefined. "
          "it can by called like this: \n\n"
          "Document('somefile.pdf').page_classifier(candidate_labels=['table_of_contents', 'credits', 'license'])"),
]

MetaDataNodes = [
    Constant(embedded_meta={}).docs("represents the metadata embedded in the file"),
    FunctionOperator(lambda x, em: dict(meta={**(x or dict()), **em})).t(dict[str, Any])
    .input(x="_meta", em="embedded_meta").out("meta").docs("Metadata of the document"),

    ## calculate some metadata values
    FunctionOperator(lambda full_text: 1 + (len(full_text) // 1000))
    .input("full_text").out("num_pages").cache().t(int)
    .docs("Number of pages in the document"),
    FunctionOperator(lambda clean_text: len(clean_text.split()))
    .input("clean_text").out("num_words").cache().t(int)
    .docs("Number of words in the document"),
    FunctionOperator(lambda spacy_sents: len(spacy_sents))
    .input("spacy_sents").out("num_sents").no_cache().t(int)
    .docs("number of sentences"),
    FunctionOperator(calculate_a_d_ratio)
    .input(ft="full_text").out("a_d_ratio").cache()
    .docs("Letter/digit ratio of the text"),
    FunctionOperator(lambda full_text: langdetect.detect(full_text))
    .input("full_text").out("language").cache()
    .default("unknown").docs(
        "Detect language of a document, return 'unknown' in case of an error"),

    ##### calculate some metadata ####
    FunctionOperator(
        lambda x: x(
            "filename",
            # "keywords",
            "document_type",
            "url",
            "path",
            "num_pages",
            "num_words",
            # "num_sents",
            "a_d_ratio",
            "language"))
    .input(x="to_dict").t(dict[str, Any])
    .out("file_meta").cache().docs(
        "Some fast-to-calculate metadata information about a document"),
]

SpacyNodes = [
    Configuration(
        spacy_model_size="md",
        spacy_model="auto",
        use_clean_text_for_spacy=True
    ).docs(
        spacy_model_size="the model size which is used for spacy text analysis. Can be:  sm,md,lg,trf.",
        spacy_model="we can also explicitly specify the spacy model we want to use.",
        use_clean_text_for_spacy="Whether pydoxtools cleans up the text before using spacy on it."
    ),
    FunctionOperator(lambda x, ct, ft: "\n\n".join(x([ElementType.Table, ElementType.Figure]).values()) if ct else ft)
    .input(x="page_templates", ct="use_clean_text_for_spacy", ft="full_text")
    .out("clean_spacy_text").t(str)
    .docs("Generate text to be used for spacy. Depending on the 'use_clean_text_for_spacy' option"
          " it will use page templates and replace complicated text structures such as tables for"
          " better text understanding."),
    SpacyOperator()
    .input(
        "language", "spacy_model",
        full_text="clean_spacy_text", model_size="spacy_model_size"
    ).out(doc="spacy_doc", nlp="spacy_nlp").cache()
    .docs("Spacy Document and Language Model for this document"),
    FunctionOperator(extract_spacy_token_vecs)
    .input("spacy_doc").out("spacy_vectors")
    .docs("Vectors for all tokens calculated by spacy"),
    FunctionOperator(get_spacy_embeddings)
    .input("spacy_nlp").out("spacy_embeddings")
    .docs("Embeddings calculated by a spacy transformer"),
    FunctionOperator(lambda spacy_doc: list(spacy_doc.sents))
    .input("spacy_doc").out("spacy_sents").t(list[str])
    .docs("List of sentences by spacy nlp framework"),
    FunctionOperator(extract_noun_chunks)
    .input("spacy_doc").out("spacy_noun_chunks")
    .docs("exracts nounchunks from spacy. Will not be cached because it is all"
          "in the spacy doc already"),
]

KnowledgeGraphNodes = [

    # TODO: combine entities with coreferences
    EntityExtractor().cache()
    .input("spacy_doc").out("entities").cache()
    .docs("Extract entities from text"),
    # TODO: try to implement as much as possible from the constants below for all documentypes
    #       summary, urls, main_image, keywords, final_url, pdf_links, schemadata, tables_df
    # TODO: implement summarizer based on textrank
    Alias(url="source").docs("Url of this document"),
    ExtractRelationships()
    .input("spacy_doc")
    .out("semantic_relations").cache()
    .docs("Extract relations from text for building a knowledge graph"),
    Configuration(
        coreference_method="fast",
        graph_debug_context_size=0
    ).docs("can be 'fast' or 'accurate'"),
    CoreferenceResolution().input("spacy_doc", method="coreference_method")
    .out("coreferences").cache(allow_disk_cache=True)
    .docs("Resolve coreferences in the text"),

    FunctionOperator(lambda x, t, do, ps, cts, m: build_document_graph(
        semantic_relations=x, coreferences=t, document_objects=do, page_set=ps,
        cts=cts, meta=m
    )).input(x="semantic_relations", t="coreferences", do="document_objects", ps="page_set",
             cts="graph_debug_context_size", m="meta")
    .out("document_graph").cache(allow_disk_cache=True)
    .t(document_graph=networkx.DiGraph)
    .docs("Builds a [networkx graph](https://networkx.org/documentation/stable/reference/classes/digraph.html) "
          "from the relations and coreferences"),

    Alias(DG="document_graph")
]

VectorizationNodes = [
    ########### VECTORIZATION (SPACY) ##########
    Alias(sents="spacy_sents").docs("Sentences of this document"),
    Alias(noun_chunks="spacy_noun_chunks").docs("Noun chunks of this documents"),

    FunctionOperator(lambda x: x.vector)
    .input(x="spacy_doc").out("vector").cache()
    .docs("Embeddings from spacy"),
    # TODO: make this configurable.. either we want
    #       to use spacy for this or we would rather have a huggingface
    #       model doing this...
    FunctionOperator(
        lambda x: dict(
            sent_vecs=np.array([e.vector for e in x]),
            sent_ids=list(range(len(x)))))
    .input(x="sents").out("sent_vecs", "sent_ids").cache()
    .docs("Vectors for sentences & sentence_ids"),
    FunctionOperator(
        lambda x: dict(
            noun_vecs=np.array([e.vector for e in x]),
            noun_ids=list(range(len(x)))))
    .input(x="noun_chunks").out("noun_vecs", "noun_ids").cache()
    .docs(
        "Vectors for nouns and corresponding noun ids in order to find them in the spacy document"),

    ########### VECTORIZATION (Huggingface) ##########
    Alias(sents="spacy_sents"),
    Alias(noun_chunks="spacy_noun_chunks"),
    Configuration(
        vectorizer_model="sentence-transformers/all-MiniLM-L6-v2",
        vectorizer_only_tokenizer=False,
        vectorizer_overlap_ratio=0.1
    ).docs("Choose the embeddings model (huggingface-style) and if we want"
           "to do the vectorization using only the tokenizer. Using only the"
           "tokenizer is MUCH faster and uses lower CPU than creating actual"
           "contextual embeddings using the model. BUt is also lower quality"
           "because it lacks the context."),
    FunctionOperator(
        lambda m, t, o: lambda txt: nlp_utils.calculate_string_embeddings(
            text=txt, model_id=m, only_tokenizer=t, overlap_ratio=o)[0].mean(0)
    ).input(m="vectorizer_model", t="vectorizer_only_tokenizer", o="vectorizer_overlap_ratio")
    .out("vectorizer").cache()
    .docs("Get the vectorizer function used for this document for an arbitrary text"),
    FunctionOperator(
        lambda x, m, t, o: nlp_utils.calculate_string_embeddings(
            text=x, model_id=m, only_tokenizer=t, overlap_ratio=o)
    ).input(x="full_text", m="vectorizer_model",
            t="vectorizer_only_tokenizer", o="vectorizer_overlap_ratio")
    .out("vec_res").cache()
    .docs("Calculate context-based vectors (embeddings) for the entire text"),
    FunctionOperator(lambda x: dict(emb=x[0], tok=x[1]))
    .input(x="vec_res").out(emb="tok_embeddings", tok="tokens").no_cache()
    .docs("Get the tokenized text"),
    FunctionOperator[list[float]](lambda x: x.mean(0))
    .input(x="tok_embeddings").out("embedding").cache()
    .docs("Get a vector (embedding) for the entire text by taking the mean of the "
          "contextual embeddings of all tokens"),
]

IndexNodes = [
    ########### SEGMENT_INDEX ##########
    Configuration(
        min_size_text_segment=256,
        max_size_text_segment=512,
        text_segment_overlap=0.3,
        max_text_segment_num=100,
    ).docs("controls the text segmentation for knowledge bases"
           "overlap is only relevant for large text segmenets that need to"
           "be split up into smaller pieces."),
    TextPieceSplitter()
    .input(min_size="min_size_text_segment", max_size="max_size_text_segment",
           large_segment_overlap="text_segment_overlap", full_text="full_text",
           max_text_segment_num="max_text_segment_num")
    .out("text_segments").cache()
    .docs("Split the text into segments"),
    # TODO: we would like to have the context-based vectors so we should
    #       calculate this "backwards" from the vectors for the entire text
    #       and not for each individual segment...
    ElementWiseOperator(calculate_string_embeddings, return_iterator=False)
    .input(elements="text_segments",
           model_id="vectorizer_model",
           only_tokenizer="vectorizer_only_tokenizer")
    .out("text_segment_vec_res").cache()
    .docs("Calculate the embeddings for each text segment"),
    FunctionOperator(lambda x: np.array([r[0].mean(0) for r in x]))
    .input(x="text_segment_vec_res").out("text_segment_vecs").cache()
    .docs("Get the embeddings for individual text segments"),
    FunctionOperator(lambda x: np.array(range(len(x)))).input(x="text_segments")
    .out("text_segment_ids").cache()
    .docs("Get the a list of ids for individual text segments"),
    IndexExtractor()
    .input(vecs="text_segment_vecs", ids="text_segment_ids").out("text_segment_index")
    .cache().docs("Create an index for the text segments"),
    KnnQuery().input(index="text_segment_index", idx_values="text_segments",
                     vectorizer="vectorizer")
    .out("segment_query").cache()
    .docs("Create a query function for the text segments which can be used to do nearest-neighbor queries"),

    ########### NOUN_INDEX #############
    IndexExtractor()
    .input(vecs="noun_vecs", ids="noun_ids").out("noun_index").cache()
    .docs("Create an index for the nouns"),
    FunctionOperator(lambda spacy_nlp: lambda x: spacy_nlp(x).vector)
    .input("spacy_nlp").out("spacy_vectorizer").cache()
    .docs("Create a vectorizer function from spacy library."),
    KnnQuery().input(index="noun_index", idx_values="noun_chunks",
                     vectorizer="spacy_vectorizer")
    .out("noun_query").cache()
    .docs("Create a query function for the nouns which can be used to do nearest-neighbor queries"),
    SimilarityGraph().input(index_query_func="noun_query", source="noun_chunks")
    .out("noun_graph").cache()
    .docs("Create a graph of similar nouns"),
    Configuration(top_k_text_rank_keywords=5),
    TextrankOperator()
    .input(top_k="top_k_text_rank_keywords", G="noun_graph").out("textrank_keywords").cache()
    .docs("Extract keywords from the graph of similar nouns"),
    # TODO: we will probably get better keywords if we first get the most important sentences or
    #       a summary and then exract keywords from there :).
    Alias(keywords="textrank_keywords"),
    ########### END NOUN_INDEX ###########

    ########### SENTENCE_INDEX ###########
    IndexExtractor()
    .input(vecs="sent_vecs", ids="sent_ids").out("sent_index").cache()
    .docs("Create an index for the sentences"),
    KnnQuery().input(index="sent_index", idx_values="spacy_sents",
                     vectorizer="spacy_vectorizer")
    .out("sent_query").cache()
    .docs("Create a query function for the sentences which can be used to do nearest-neighbor queries"),
    SimilarityGraph().input(index_query_func="sent_query", source="spacy_sents")
    .out("sent_graph").cache()
    .docs("Create a graph of similar sentences"),
    Configuration(top_k_text_rank_sentences=5)
    .docs("controls the number of most important sentences that are extracted from the text."),
    TextrankOperator()
    .input(top_k="top_k_text_rank_sentences", G="sent_graph").out("textrank_sents").cache()
    .docs("Extract the most important sentences from the graph of similar sentences"),
]

LLMNodes = [
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
    # FunctionOperator(lambda x, y: x(y))
    # .pipe(x="summary_func", y="full_text").out("summary").cache(),
    FunctionOperator(lambda x, m, to, ml: summarize_long_text(
        x, m, token_overlap=to, max_len=ml
    )).input(
        x="clean_text", m="summarizer_model",
        to="summarizer_token_overlap",
        ml="summarizer_max_text_len"
    ).out("slow_summary").cache()
    .docs("Summarize the text using the Huggingface summarization pipeline"),

    ########### QaM machine #############
    # TODO: make sure we can set the model that we want to use dynamically!
    Configuration(qam_model_id='deepset/minilm-uncased-squad2'),
    QamExtractor()
    .input(property_dict="to_dict", trf_model_id="qam_model_id").out("answers").cache()
    .docs("Extract answers from the text using the Huggingface question answering pipeline"),

    ########### Chat AI ##################
    Configuration(chat_model_id="gpt-3.5-turbo").docs(
        "In order to use openai-chatgpt, you can use 'gpt-3.5-turbo' or 'gpt-4'."
        "Additionally, we support models used by gpt4all library which"
        "can be run locally and most are available for commercial purposes. "
        f"Currently available models are: {extract_nlpchat.gpt4_models()}"
    ),
    extract_nlpchat.LLMChat().input(property_dict="to_dict", model_id="chat_model_id")
    .out("chat_answers").cache()
    .docs("Extract answers from the text using OpenAI Chat GPT and other models."),
]

TextStructureNodes = [
    FunctionOperator(extract_text_elements)
    .input(text="full_text").out("elements")
    .docs("extracts a list of document objects such as tables, text boxes, figures, etc.")
    .cache(),
    FunctionOperator(lambda els, ltbs: {
        id: el for id, el in enumerate(els + ltbs) if el != ElementType.Text
    }).t(dict[int, DocumentElement])
    .input(els="elements", ltbs="labeled_text_boxes").out("document_objects")
    .docs("output a list of document elements which can be referenced by id"),
    Alias(do="document_objects"),
    FunctionOperator(lambda x: [tb for tb in x if tb.type == ElementType.TextBox])
    .input(x="elements").out("text_box_elements").t(list[str]).cache()
    .docs("Text boxes extracted as a pandas Dataframe with some additional metadata"),
    DocumentElementFilter(element_type=ElementType.Header)
    .input("elements").out("headers").cache()
    .docs("Extracts the headers from the document"),
    # get tables only from document_objects this will also give as the correct reference!
    FunctionOperator(lambda do: {idx: t for idx, t in do.items() if t.type == ElementType.Table})
    .t(dict[int, DocumentElement])
    .input(do="document_objects").out("tables").cache()
    .docs("Extracts the tables from the document as a dataframe"),
    FunctionOperator(lambda x: [pd.DataFrame(t.obj) for idx, t in x.items()]).t(list[pd.DataFrame])
    .input(x="tables").out("tables_df").cache(),
    DocumentElementFilter(element_type=ElementType.List)
    .input("elements").out("lists").cache()
    .docs("Extracts the lists from the document"),
    DocumentElementFilter(element_type=ElementType.Text)
    .input("elements").out("line_elements").cache()
    .docs("Filters the document elements and only keeps the text elements"),
    DocumentElementFilter(element_type=ElementType.Graphic)
    .input("elements").out("graphic_elements").cache()
    .docs("Filters the document elements and only keeps the graphic elements"),
    DocumentElementFilter(element_type=ElementType.Image)
    .input("elements").out("image_elements").cache()
    .docs("Filters the document elements and only keeps the image elements"),

    ##### generalized pages ########
    # TODO: extract the correct page number for all documents
    #     # TODO: replace with a generalized page template generator...
    extract_textstructure.PageTemplateGenerator()
    .input("document_objects").out("page_templates").cache()
    .docs("generates a text page with table & figure hints"),

    FunctionOperator(lambda pt, ps: "".join(f"\n\n-------- {p} --------\n\n" + pt()[p] for p in ps))
    .input(pt="page_templates", ps="page_set").out("page_templates_str").cache()
    .t(str)
    .docs("Outputs a nice text version of the documents with annotated document objects"
          " such as page numbers, tables, figures, etc."),
    FunctionOperator(lambda pt, ps: "".join(pt(["TextBox"], include="True")[p] for p in ps))
    .input(pt="page_templates", ps="page_set").out("page_templates_str_minimal").cache()
    .t(str)
]

document_operators = {
    # .pdf
    "application/pdf": PDFNodes,
    # .html
    "text/html": HTMLNodes,
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
    # wikipedia data dump
    "mediawiki": ["pandoc"],
    # pandoc document conversion pipeline
    "pandoc": PandocNodes,
    # standard image pipeline
    "image": OCRNodes,
    # the first base doc types have priority over the last ones
    # so here .png > image > .pdf
    'PIL.Image.Image': PILImageNodes,
    'image/png': ["image", "application/pdf"],
    'image/jpeg': ["image", "application/pdf"],
    'image/tiff': ["image", "application/pdf"],
    "application/x-yaml": [
        "<class 'dict'>",
        Alias(full_text="raw_content"),
        FunctionOperator(lambda x: yaml.unsafe_load(x))
        .input(x="full_text").out("data").cache()
        .docs("Load yaml data from a string"),
        # TODO: we might need to have a special "result" message, that we
        #       pass around....
    ],
    # simple dictionary with arbitrary data from python
    "<class 'dict'>": [  # pipeline to handle data based documents
        FunctionOperator(lambda x: yaml.dump(list_utils.deep_str_convert(x))).t(str)
        .input(x="data").out("full_text").cache()
        .docs("Dump dict data to a yaml-like string"),
        DictSelector()
        .input(selectable="data").out("data_sel").cache().docs(
            "select values by key from source data in Document"),
        # TODO: use a document_element creation function for this...
        FunctionOperator(lambda x: [DocumentElement(
            type=ElementType.TextBox,
            text=str(k) + ": " + str(v)
        ) for k, v in list_utils.flatten_dict(x).items()])
        .input(x="data").out("text_box_elements").cache().t(list[str])
        .docs("Create a dataframe from a dictionary. TODO: this is not working correctly, it should "
              "create a list of [][pydoxtools.document_base.DocumentELements]"),
        # TODO: make this resemble pytho dicts?
        #       by returning keys, values, items as normal functions?
        FunctionOperator(lambda x: x.keys())
        .input(x="data").out("keys").no_cache()
        .docs("Get the keys of the dictionary"),
        FunctionOperator(lambda x: x.values())
        .input(x="data").out("values").no_cache()
        .docs("Get the values of the dictionary"),
        FunctionOperator(lambda x: x.items())
        .input(x="data").out("items").no_cache()
        .docs("Get the items of the dictionary"),
    ],
    "<class 'list'>": [
        FunctionOperator(lambda x: yaml.dump(list_utils.deep_str_convert(x))).t(str)
        .input(x="data").out("full_text").cache()
        .docs("Dump list data to a yaml-like string"),
        FunctionOperator(lambda x: [DocumentElement(
            type=ElementType.TextBox,
            text=str(list_utils.deep_str_convert(v))
        ) for v in x])
        .input(x="data").out("text_box_elements").cache().t(list[str]),
    ],
    # TODO: json, csv etc...
    # TODO: pptx, odp etc...
    "*": [
        Alias(data="raw_content").t(Any).docs("The unprocessed data."),
        Constant(page_set={0}),
        FunctionOperator(lambda x: force_decode(x)).t(str)
        .input(x="raw_content").out("full_text").docs(
            "Full text as a string value"),
        Alias(clean_text="full_text").t(str),
        # TODO: replace this with a real, generic table detection
        #       e.g. running the text through pandoc or scan for html tables
        Constant(tables_df=[]),
        # TODO: define datatype correctly
        FunctionOperator(lambda tables_df: [df.to_dict('index') for df in tables_df]).cache()
        .input("tables_df").out("tables_dict").t(list[dict])
        .docs("List of Table"),
        Alias(tables="tables_dict"),

        *TextStructureNodes,
        *ClassifierNodes,
        *MetaDataNodes,
        *SpacyNodes,
        *KnowledgeGraphNodes,
        *VectorizationNodes,
        *IndexNodes,
        *LLMNodes
    ]
}


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

## Customizing the Document Pipeline:

The extraction pipeline can be partially overwritten or completely replaced to customize the
document processing. To customize the pipeline, it's recommended to use the basic document
pipeline defined in `pydoxtools.Document` as a starting point and only overwrite parts as needed.

Inherited classes can override any part of the graph. To exchange, override, extend or introduce
extraction pipelines for specific file types (including the generic one: "*"), such as *.html,
*.pdf, *.txt, etc., follow the example below.

### Rules for customizing the extraction pipeline:

- The pipeline is defined as a dictionary of several lists of [pydoxtools.operator_base.Operator][operator]
  nodes.
- Each [pydoxtools.operator_base.Operator][] defines a set of output & input valus through
  the `out` and `input` methods.
  The `input` method takes a dictionary
  or list of input values and the `out` method takes a dictionary or list of output values.
- Operator nodes are configured through method chaining.
- Arguments can be overwritten by a new pipeline in inherited
  documents or document types higher up in the hierarchy. The argument precedence is as follows:

  ```
  python-class-member < extractor-graph-function < configuration
  ```

- the different lists in the dictionary represent a "hierarchy" of pipelines which
  can be combined in different ways by referencing each other. For example the "png" pipeline
  references the "image" pipeline which in turn references the "pdf" pipeline. All pipelines
  fall back to the "*" pipeline which is the most generic one.

  The way this looks is like this:

    _operators = {
            # .pdf-specific pipeline
            "application/pdf": [*PDFNodes],
            # image specific pipeline (does OCR)
            "image": [*OCRNodes],
            # .png-specific pipeline
            ".png": ["image", "application/pdf"],
            # base pipeline
            "*": [*BaseNodes],
            }

    Here, the "image" pipeline overwrites the "application/pdf" pipeline for .png files. The "application/pdf"
    pipeline on the other hand overwrites the "*" pipeline for .pdf files. The "*" pipeline doesn't not need
    to be specified, as it will always be the fallback pipeline. This way it is possible
    to dynamically adapt a pipeline to different types of input data. In the document pipeline
    this is used to dynamically adapt the pipeline to different file types.

    When customizing, it is possible to derive a new class from [pydoxtools.Document][] and partially
    overwrite its hierarchy for your purposes. For example, if you want to add a new pipeline for Component
    extraction one could do something like the following: This will add a few more nodes
    to the generic pipeline in order to extract product information from documents. This would
    now already work for all the document types defined in the base [pydoxtools.Document][] class!

    ```python
    class DocumentX(pydoxtools.Document):
    _operators = {
        "*": [
            FunctionOperator(get_products_from_pages).input("page_templates", pages="page_set")
            .out(product_information="product_information").cache(),
            FunctionOperator(lambda tables: [componardo.spec_utils.table2specs(t) for t in tables])
            .input("tables").out("raw_specs")
            .cache().t(list[componardo.spec_utils.Specification])
            .docs("transform tables into a list of specs"),
            FunctionOperator(lambda x: [ComponentExtractor([x]).component])
            .t(componardo.extract_product.ComponentX)
            .input(x="document_extract").out("products").cache()
            .docs("Extract products from Documents"),
        ]
    }
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

    # compose a list of operators into a dynamic pipeline which changes
    # depending on the file type
    _operators = document_operators

    def __init__(
            self,
            fobj: str | bytes | Path | IO | dict | list | set = None,
            # TODO: only use Path and declare the variables using pydantic we'll also get validation
            source: str | Path = None,
            meta: dict[str, str] = None,
            document_type: str = "auto",
            page_numbers: list[int] = None,
            max_pages: int = None,
            configuration: dict = None,
            **kwargs,
    ):
        """Initialize a Document instance.

        Either fobj or source are required. They can both be given. If either of them
        isn't specified the other one is inferred automatically.

        document_type, page_number and max_pages are also not required, but can be used to override
        the default behaviour. specifically document_tgiype can be used manually specify
        the pipeline that should be used.

        Args:
            fobj:
                The file object or data to load. Depending on the type of object passed:
                - If a string or bytes object: the object itself is the document. IN case of a bytes
                     object, the source helps in determining the filetype through file endings.
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

            meta:
                Optionally set document metadata, which can be very useful for downstream
                tasks like building an index.

            page_numbers:
                A list of specific pages to extract from the document (e.g., in a PDF).

            max_pages:
                The maximum number of pages to extract to protect resources.

            configuration:
                configuration dictionary for the pipeline
        """

        doc_configuration = {}
        doc_configuration.update(configuration or {})
        doc_configuration.update(kwargs)
        super().__init__(**doc_configuration)

        # TODO: move this code into its own little extractor...
        #       can also be made better with pydantic ;).

        if not (source or fobj):
            raise DocumentInitializationError(
                "Either 'source' or 'fobj' are required for initialzation of Document Class")

        self._fobj = fobj  # file or data object
        self._source = source
        self._document_type = document_type  # override pipeline selection
        self._meta = meta or {}
        self._page_numbers = page_numbers
        self._max_pages = max_pages

        if ((self._document_type == "auto" and self.magic_library_available()) or
                self._document_type not in ["auto", "string", str(dict)]):
            try:
                # TODO: this is an unfortunate position..  somehow should refactor documenttype detection,
                #       fileloader and this here to download urls
                if is_url(fobj):
                    response = requests.get(fobj)
                    self._fobj = response.content
            except:
                pass

    @functools.lru_cache
    def _pipeline_key(self):
        return (self.__class__.__name__, str(self._configuration), self._fobj, self._source,
                self._document_type, self._page_numbers, self._max_pages)

    @cached_property
    def fobj(self) -> bytes | str | Path | IO | dict | list | set:
        if self._fobj:
            return self._fobj
        else:
            return self._source

    @cached_property
    def source(self) -> str:
        return self._source or self.path

    @property
    def filename(self) -> str | None:
        """return filename or some other identifier of a file"""
        _, filepath, _ = self.document_type_detection()
        if isinstance(filepath, Path):
            return filepath.name
        else:
            return filepath or "<unknown>"

    @property
    def path(self):
        _, filepath, _ = self.document_type_detection()
        return filepath

    @property
    def raw_content(self):
        _, _, buffer = self.document_type_detection()
        return buffer

    @property
    def document_type(self):
        """This has to be done in a member function and not
        in the pipeline, because the selection of the pipeline depends on this..."""
        # document type was overriden
        if self._document_type != "auto":
            return self._document_type
        else:
            doc_type, _, _ = self.document_type_detection()
            return doc_type

    @staticmethod
    @functools.lru_cache  # make sure we only run it once..
    def magic_library_available():
        try:
            import magic
        except ImportError:
            logger.warning("libmagic was not found on your system! falling back"
                           "to filetype detection by file-ending only. In order"
                           "to have a more robust file type detection it would be"
                           "advisable to install this library.")
            magic = False
        return magic

    @functools.lru_cache
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
        detected_filepath: Path | None = None
        mimetype = "unknown"  # use this as a standard mimetype

        magic = Document.magic_library_available()

        _fobj = self._fobj or self._source

        # if it works... now, check filetype by analyzing
        # the content of a file and load the file itself into
        # a buffer
        if isinstance(_fobj, (Path, str)):
            # if it's not a file, we take it as a string, but can be overwritten
            # using the document_type variable during initialization
            buffer = self._fobj
            try:
                if self._document_type == "string":
                    mimetype = "string"
                elif Path(_fobj).is_file():  # check if we have an actual file here
                    detected_filepath = Path(_fobj)
                    buffer = load_raw_file_content(detected_filepath)
                    mimetype, _ = mimetypes.guess_type(detected_filepath, strict=False)
                    if mimetype is None:
                        if magic:
                            mimetype = magic.from_file(_fobj, mime=True)
                        else:
                            logger.warning("install magic to guess file type!")
                else:
                    mimetype = "string"
            except OSError:  # could happen if we try to use a string as filename and that is too long
                mimetype = "string"

        elif isinstance(self._fobj, bytes):
            buffer = self._fobj
            if magic:
                mimetype = magic.from_buffer(buffer, mime=True)
            else:
                logger.warning(f"no filetype specified for {self._source}")
        elif isinstance(self._fobj, io.IOBase):  # might be a streaming object
            # TODO: there is probably a better way to check for many file io objects...
            buffer = self._fobj.read()
            self._fobj.seek(0)  # reset pointer for downstream tasks
            try:
                detected_filepath = Path(self._fobj.name)
            except:
                pass
            if magic:
                mimetype = magic.from_buffer(buffer, mime=True)
            elif detected_filepath:
                mimetype, _ = mimetypes.guess_type(detected_filepath, strict=False)
        elif isinstance(self._fobj, (dict, list, set)):
            mimetype = str(type(self._fobj))
            buffer = self._fobj
        elif isinstance(self._fobj, PIL.Image.Image):
            b = io.BytesIO()
            self._fobj.save(b, 'PNG')
            buffer = b.getvalue()
            mimetype = 'PIL.Image.Image'
        elif isinstance(self._fobj, tempfile.SpooledTemporaryFile):
            detected_filepath = self._source
            buffer = self._fobj.read()
            if self._document_type == "auto":
                mimetype = magic.from_buffer(buffer, mime=True)
            else:
                mimetype = self._document_type
        else:
            mimetype = str(type(self._fobj))
            buffer = self._fobj

        if mimetype == "unknown" and self._document_type != "auto":
            mimetype = self._document_type

        if mimetype == "unknown" and self._source:
            mimetype, encoding = mimetypes.guess_type(self._source, strict=False)

        # do some more checks on problems that we have encountered when using "magic"
        if mimetype == "application/json":
            try:
                json.loads(buffer)
            except json.decoder.JSONDecodeError:
                mimetype = "text/plain"

        # do some more checks for several special cases hat we found using
        # the file-ending as a first check
        # the file ending becomes a lot more important in this case!
        if mimetype == "text/html":
            mimetype = is_html_or_xml(force_decode(buffer))
        if mimetype in ("application/xml", "unknown"):
            mimetype = detect_xml_type(force_decode(buffer))
        if mimetype in ("text/plain", "unknown"):
            if contains_markdown(force_decode(buffer)):
                mimetype = "text/markdown"

        # do a mapping to standardize file type detection a bit more:
        mimetype = {
            'application/rtf': 'text/rtf',
            None: 'UNKNOWN'  # TODO: try to not make everything that we "don't know" a text-file...
        }.get(mimetype, mimetype)

        return mimetype, detected_filepath, buffer

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
        if isinstance(self._source, (str, bytes)):
            return f"{self.__module__}.{self.__class__.__name__}(source={self.source[-10:]})"
        else:
            return f"{self.__module__}.{self.__class__.__name__}(source={self.source})"

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


class DocumentBagCreator(Operator):
    """
    Basically it creates a Documentbag from two sets of
    on documents in a dask bag and then creates a new DocumentBag from that. This
    works similar to pandas dataframes and series. But with documents
    as a basic datatype. And apply functions are also required to
    produce data which can be used as a document again (which is a lot).
    """

    def __call__(self,
                 apply_map_func: Callable,
                 configuration: dict[str, Any],
                 forgiving_extracts: bool,
                 ) -> Callable[..., "DocumentBag"]:
        def doc_creator_func(
                new_document: Callable[[Document], Any] | str | list[str],
                document_metas: Callable[[Document], Any] | str | list[str] = None,
        ) -> DocumentBag:

            def extract(d):
                if callable(new_document):
                    fobj = new_document(d)
                else:
                    list_doc_mapping = list_utils.ensure_list(new_document)
                    fobj = d.to_dict(*list_doc_mapping)

                fobj = remove_list_from_lonely_object(fobj)

                if document_metas:
                    if callable(document_metas):
                        meta_dict = document_metas(d)
                    else:
                        list_meta_mapping = list_utils.ensure_list(document_metas)
                        meta_dict = d.to_dict(*list_meta_mapping)
                    if len(meta_dict) == 1:
                        meta_dict = next(iter(meta_dict.values()))
                else:
                    meta_dict = d.meta

                return fobj, meta_dict

            def document_mapping(d: Document):
                if forgiving_extracts:
                    try:
                        fobj, meta_dict = extract(d)
                    except Exception as err:
                        fobj = str(err)
                        meta_dict = {"Error": str(err)}
                else:
                    fobj, meta_dict = extract(d)

                new_doc = Document(
                    fobj, source=d.source, meta=meta_dict, configuration=d.configuration)

                return new_doc

            new_documents_bag = apply_map_func(document_mapping)
            db = DocumentBag(new_documents_bag, configuration=configuration)
            return db

        return doc_creator_func


class DocumentBagExplode(Operator):
    """
    This operator wil take a DocumentBag and check whether the documents
    inside consist of list-data and then split them up into separate documents while
    flattening the list of lists at the same time.
    """

    def __call__(
            self, apply_map_func: Callable,
            configuration: dict[str, Any]
    ) -> Callable[..., "DocumentBag"]:
        def split_func(d: Document) -> list[Document]:
            if d.document_type == "<class 'list'>":
                new_docs = []
                for obj in d.fobj:
                    new_docs.append(
                        Document(obj, source=d.source,
                                 document_type="string", meta=d.meta,
                                 configuration=d.configuration
                                 ))
                return new_docs
            else:
                return [d]

        new_documents_bag = apply_map_func(split_func)
        dask_bag_function = "flatten"  # e.g. "bag.flatten" the resulting object...
        new_documents = getattr(new_documents_bag, dask_bag_function)()
        db = DocumentBag(new_documents, configuration=configuration)
        return db


class DatabaseSource(pydantic.BaseModel):
    connection_string: str
    sql: str
    index_column: str


class DocumentBag(Pipeline):
    """
    This class is a work-in-progress (WIP), use with caution.

    The DocumentBag class loads and processes a set of documents using a pipeline.
    It leverages Dask bags for efficient memory usage and large-scale computations on documents.

    Notes:
        - Dask bags documentation can be found [here](https://docs.dask.org/en/stable/bag.html).
        - Dask dataframes can be used for downstream calculations.
        - This class helps scale LLM & AI inference to larger workloads.
        - It uses iterative Dask bags & dataframes to avoid out-of-memory issues.

    Rationale:
        This function is needed to create and process new document bags, instead of using Dask bags directly
        with arbitrary data. It reduces boilerplate code for creating new documents and traceable datasources.
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
    _operators = {
        str(DatabaseSource): [
            str(Bag),  # DatabaseSource will eventually be turned into a bag of documents
            FunctionOperator(lambda x: x.dict())
            .input(x="source")
            .out("sql", "connection_string", "index_column").cache(),
            Configuration(bytes_per_chunk="256 MiB"),
            SQLTableLoader()
            .input("sql", "connection_string", "index_column", "bytes_per_chunk").out("dataframe").cache(),
            FunctionOperator(lambda x: x.to_bag(index=True, format="dict"))
            .input(x="dataframe")
            .out("bag").cache(),
            dask_operators.BagMapOperator(lambda y, c: Document(
                y, source=str(y.get('index', None)), configuration=c))
            .input(dask_bag="bag", c="doc_configuration").out("docs").cache().docs(
                "Create a dask bag of one data document for each row of the source table"),
        ],
        str(Bag): [
            # TODO: accept arbitrary bags
            # Alias(bag="source"),
            Alias(docs="source"),
            FunctionOperator(lambda x: x.take)
            .input(x="docs").out("take").cache(),
            FunctionOperator(lambda x: x.compute)
            .input(x="docs").out("compute").cache(),
        ],
        # TODO: add "fast, slow, medium" pipelines..  e.g. with keywords extraction as fast
        #       etc...
        # TODO: add a pipeline to add a summarizing description whats in every directory
        str(list): [
            str(Path), str(Bag),
            # load all paths into a bag
            FunctionOperator(lambda x: dask.bag.from_sequence((Path(p) for p in x), partition_size=10))
            .input(x="source").out("bag"),
            # filter for actual files
            dask_operators.BagFilterOperator(lambda it: it.is_file())
            .input(dask_bag="bag").out("file_path_list"),
            # filter for directories
            dask_operators.BagFilterOperator(lambda it: it.is_dir())
            .input(dask_bag="bag").out("dir_list")
        ],
        str(Path): [
            str(Bag),
            # TODO:  add a string filter which can be used to filte paths & db entries
            #        and is simply a bit more generalized ;)
            # TODO: make this all an iterable...  maybe even using dask..
            Alias(root_path="source"),
            # TODO: make PathLoader an iterator!
            PathLoader()
            .input(directory="root_path", exclude="_exclude")
            .out("paths").cache(),
            FunctionOperator(lambda x: x(max_depth=10, mode="files"))
            .input(x="paths")
            .out("file_path_list").cache(),
            FunctionOperator(lambda x: x(max_depth=10, mode="dirs"))
            .input(x="paths")
            .out("dir_list").cache(),
            FunctionOperator(lambda x: dask.bag.from_sequence(x, partition_size=10))
            .input(x="file_path_list").out("bag").cache().docs(
                "create a dask bag with all the filepaths in it"),
            dask_operators.BagMapOperator(lambda x, c: Document(x, configuration=c))
            .input(dask_bag="bag", c="doc_configuration").out("docs").cache().docs(
                "create a bag with one document for each file that was found"
                "From this point we can hand off the logic to str(Bag) pipeline."),
            # TODO: extract some metadata about files etc..  as a new DocumentBag or as a Document?!
            # DataMerger()
            # .pipe(root_dir="_source")
            # .out(joint_data="meta_data").cache()
        ],
        "*": [
            Configuration(doc_configuration=dict()).docs(
                "We can pass through a configuration object to Documents that are"
                " created in our document bag. Any setting that is supported"
                " by Document can be specified here."),
            Configuration(forgiving_extracts=False).docs(
                "When enabled, if we execute certain batch operations on our"
                " document bag, this will not stop the extraction, but rather put an error message"
                " in the document."
            ),
            Constant(_stats=[]),
            Configuration(verbosity=None),
            dask_operators.BagPropertyExtractor()
            .input("verbosity", dask_bag="docs", forgiving_extracts="forgiving_extracts", stats="_stats")
            .out("get_dicts").cache(allow_disk_cache=False),
            Alias(d="get_dicts"),
            DocumentBagMap()
            .input("verbosity", dask_bag="docs", forgiving_extracts="forgiving_extracts",
                   stats="_stats")
            .out("bag_apply"),
            DocumentBagCreator()
            .input("configuration", apply_map_func="bag_apply", forgiving_extracts="forgiving_extracts")
            .out("apply"),
            DocumentBagExplode()
            .input("configuration", apply_map_func="bag_apply")
            .out("exploded").docs(""),
            Alias(e="exploded"),
            FunctionOperator(lambda x: pd.DataFrame(x).sum())
            .input(x="_stats").out("stats").no_cache()
            .docs("gather a number of statistics from documents as a pandas dataframe"),
            FunctionOperator(
                lambda x: lambda *args, **kwargs: Document(
                    *args, **kwargs, configuration=x))
            .input(x="doc_configuration").out("Document").docs(
                "Get a factory for pre-configured documents. Can be called just like"
                " [pydoxtools.Document][] class, but automatically gets assigned the"
                " same configuration as all Documents in this bag"),
            FunctionOperator(lambda doc: lambda text: doc(text).embedding)
            .input(doc="Document").out("vectorizer").cache()
            .docs("vectorizes a query, using the document configuration of the Documentbag"
                  " to determine which model to use."),
        ]
    }

    #       bag & documentbag

    def __init__(
            self,
            source: str | Path | DatabaseSource | Bag | list[str | Path],  # can be sql information,
            pipeline: str = None,
            exclude: list[str] = None,
            max_documents: int = None,
            configuration: dict = None,
            **kwargs
    ):
        # TODO: add "overwrite" parameter where we basically say that DocumentBag should
        #       automatically overwrite configurations of documents it creates
        docbag_configuration = {}
        docbag_configuration.update(configuration or {})
        docbag_configuration.update(kwargs)
        super().__init__(**docbag_configuration)
        self._exclude = tuple(exclude or [])
        # TODO: _pipeline isn't used yet
        self._pipeline = pipeline or "directory"
        self._max_documents = max_documents

        self._source = source
        if isinstance(source, str):
            if Path(source).exists():
                self._source = Path(source)

    def _pipeline_key(self):
        return (self.__class__.__name__, str(self._configuration), self._exclude, self._pipeline, self._max_documents,
                self._source)

    @cached_property
    def source(self):
        return self._source

    @cached_property
    def pipeline_chooser(self) -> str:
        if isinstance(self._source, Path):
            return str(Path)
        else:
            return str(type(self._source))

    def __repr__(self):
        """
        Returns:
            str: A string representation of the instance.
        """
        if isinstance(self._source, str | bytes):
            return f"{self.__module__}.{self.__class__.__name__}({self.source[:10]})"
        else:
            return f"{self.__module__}.{self.__class__.__name__}({self.source})"
