"""
This module simply gathers all operators from across the board to make them easier to
access.
"""

from pydoxtools.operators_base import Operator, Configuration
from operators_base import FunctionOperator, ElementWiseOperator, Alias, ForgivingExtractIterator,DataMerger
from operator_huggingface import HuggingfacePipeline, QamExtractor
from extract_filesystem import FileLoader, PathLoader, Path
from extract_ocr import OCRExtractor
from extract_nlpchat import OpenAIChat
from extract_pandoc import PandocLoader, PandocOperator, PandocConverter, PandocBlocks
from extract_html import HtmlExtractor
from extract_classes import LanguageExtractor, TextBlockClassifier
from extract_index import IndexExtractor, KnnQuery, TextPieceSplitter, HuggingfaceVectorizer,\
    SimilarityGraph, TextrankOperator
from extract_objects import EntityExtractor
from extract_spacy import SpacyOperator
from extract_tables import ListExtractor, TableCandidateAreasExtractor, HTMLTableExtractor, \
    Iterator2Dataframe
from extract_textstructure import DocumentElementFilter, TextBoxElementExtractor, TitleExtractor
from pdf_utils import PDFFileLoader
