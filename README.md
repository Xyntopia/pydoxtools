---
title: 'pydoxtools (Python Library)'
library name: pydoxtools
keywords: [ pydoxtools, AI, AI-Composition, ETL, pipelines, knowledge graphs ]
---


🎩✨📄 pydoxtools (Python Library) 🎩✨📄
================================================================================

![Python](https://img.shields.io/pypi/pyversions/pydoxtools.svg)
[![PyPI version](https://img.shields.io/pypi/v/pydoxtools.svg)](https://pypi.python.org/pypi/pydoxtools)
[![Mkdocs](https://img.shields.io/badge/doc-mkdocs-845ed7.svg)](https://pydoxtools.xyntopia.com)
[![GitHub discussions](https://img.shields.io/badge/discuss-online-845ef7)](https://github.com/xyntopia/pydoxtools/discussions)
[![GitHub stars](https://img.shields.io/github/stars/xyntopia/pydoxtools)](https://github.com/xyntopia/pydoxtools/stargazers)

***

[Documentation](https://pydoxtools.xyntopia.com)

*If you have any problems or questions, please create a github [issue](https://github.com/Xyntopia/pydoxtools/issues).
So that other poeple who might want to use it can see the potential solution!*

## Summary

Pydoxtools is a library that provides a sophisticated interface for reading and
writing documents, designed to work with AI models such as GPT, LLama2, and
a variety of models on Huggingface. It offers functionalities such as:b

- Pipeline management
- Integration with AI (LLMs and more) models
- low-resource (PDF) table extraction without configuration and expensive
  layout detection algorithms!
- Knowledge base extraction as a one-liner
- Document analysis and question-answering
- Support for most of todays document formats
- Vector index Creation
- Entity, address identification and more
- List and keyword extraction
- Data normalization, translation, and cleaning

The library allows for the creation of complex extraction pipelines
for batch-processing of documents by defining them as a lazily-executed graph.

## Installation

### Installing from GitHub

While pydoxtools can already be installed through pip, due to the
many updates coming in right now, it is currently recommended to use
the latest version from GitHub as follows:

    pip install -U "pydoxtools[etl,inference] @ git+https://github.com/xyntopia/pydoxtools.git"

### Installing from PyPI

Pydoxtools can also be installed through pip, which will become the recommended
method once it becomes more stable:

    pip install -U pydoxtools[etl,inference]

For loading additional file formats (docx, odt, epub), OCR and other options,
check out the additional > [Installation Options](#installation-options) <.

## 🚀 Teaser 🚀

Experience a new level of convenience and efficiency in handling
documents with Pydoxtools, and reimagine your data extraction pipelines!

In this teaser, we'll demonstrate how to create a document, extract
tables, and ask questions using AI models:

```python
import pydoxtools as pdx

# Create a document from various sources: file, string, bytestring, file-like object, or URL
doc = pdx.Document("https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf")

# List available extraction functions
print(doc.x_funcs)

# get all tables from a single document:
print(doc.tables)

# Extract the first 20 tables that we can find in a directory (this might take a while,
# make sure, to only choose a small directory for testing purposes)
docs = pdx.DocumentBag("./my_directory_with_documents", forgiving_extracts=True)
print(docs.bag_apply(["tables_df", "filename"]).take(20))

# Ask a question about the documents using a local Q&A model
print(doc.answers(["how much ram does it have?"]))
# Or only ask about the documents tables (or any other extracted information):
print(doc.answers(["how much ram does it have?"], "tables"))

# To use ChatGPT for question-answering, set the API key as an environment variable:
# OPENAI_API_KEY="sk ...."
# Then, ask questions about the document using ChatGPT
print(doc.chat_answers(["What is the target group of this document?"])[0].content)
print(doc.chat_answers(["Answer if a 5-year old would be able to follow these instructions?"])[0].content)
```

With Pydoxtools, you can easily access and process your documents, perform various extractions,
and utilize AI models for more advanced analysis.

### Supported File Formats

Pydoxtools already supports loading from a large variety of different sources:

- Documents from URLs,
- pdf, html, docx, doc, odt, markdwn, rtf, epub, mediawiki
- everything supported by [pandoc](https://pandoc.org/),
- images (png, jpg, bmp, tiff etc...),
- And some "native-python" dataformats: PIL.Image.Image, <class 'dict'>, <class 'list'>
- data formats: yaml (json in progress)
- And more!

## Some Features in More Detail

### Large Pipelines

Pydoxtools' main feature is the ability to mix LLMs and other
AI models in large, composable, and customizable pipelines.
Using pipelines comes with the slight disadvantage that it
can be more challenging to add type hints to the code.
However, using pipelines decouples all parts of your code,
allowing all operators to work independently. This
makes it easy to run the pipeline in a distributed setting for big data
and enables easy, lazy evaluation. Additionally,
mixing different LLM logics together becomes much easier.

Check out how Pydoxtools' `Document` class mixes pipelines for each individual file type:

- Every node in an ellipse can be called as an attribute of the document-analysis pipeline.
- Every execution path is lazily executed throughout the entire graph.
- Every node is cached by default (but can be turned off).
- Every piece of this pipeline can be replaced by a customized version.

As an example, consider this pipeline for *.png images from the repository,
which includes OCR, keyword extraction, vectorization, and more:

![png pipeline](http://pydoxtools.xyntopia.com/images/document_logic_.png.svg)

Pipelines can be mixed, partially overwritten, and extended, giving you a lot of possibilities
to extend and adapt the functionality for your specific use case.

To learn more about Pydoxtools' large pipelines feature, please refer to
the [documentation](http://pydoxtools.xyntopia.com/reference/#pydoxtools.document.Document).

#### Pipeline Configuration

Pipelines can be configured. For example the local model used for
question answering can be selected like this:

```python
doc = Document(fobj="./data/PFR-PR23_BAT-110__V1.00_.pdf")
.config(qam_model_id='bert-large-uncased-whole-word-masking-finetuned-squad')
```

where "qam_model_id" can be any model from huggingface for question answering.

You can get a list of configuration options like this:

```python
doc.configuration

# >> will give you something like this:
# {'spacy_model_size': 'md',
# 'spacy_model': 'auto',
# 'use_clean_text_for_spacy': True,
# 'coreference_method': 'fast',
# 'graph_debug_context_size': 0,
# 'vectorizer_model': 'sentence-transformers/all-MiniLM-L6-v2',
# 'vectorizer_only_tokenizer': False,
# 'vectorizer_overlap_ratio': 0.1,
# 'min_size_text_segment': 256,
# 'max_size_text_segment': 512,
# 'text_segment_overlap': 0.3,
# 'max_text_segment_num': 100,
# 'top_k_text_rank_keywords': 5,
# 'top_k_text_rank_sentences': 5,
# 'summarizer_model': 'sshleifer/distilbart-cnn-12-6',
# 'summarizer_token_overlap': 50,
# 'summarizer_max_text_len': 200,
# 'qam_model_id': 'deepset/minilm-uncased-squad2',
# 'chat_model_id': 'gpt-3.5-turbo',
# 'image_dpi': 216,
# 'ocr_lang': 'auto',
# 'ocr_on': True}
```

FOr more information check the documenation:

### PDF Table Extraction Algorithms

The library features its own sophisticated Table extraction algorithm which is benchmarked
against a large pdf table dataset. In contrast to how most "classical" table extraction
algorithms work, it doesn't require:

- extensive configuration
- no expensive deep neural networks for table area recognition which need a GPU and
  a lot of memory/CPU requirements

This makes it possible to run analysis on PDF files with pydoxtools on CPU with
very limited resources!

### TODO: Describe more of the features here...

## Use Cases

- create new documents from unstructured information
- analyze documents using any model from huggingface
- analyze documents using a custom model
- download a pdf from URL
- generate document keywords
- extract tables
- download document from URL "manually" and then feed to document
- extract addresses
- extract addresses and use this information for the qam
- ingest documents into a vector db

## Installation Options

If you simply want to get going, you can install the following libraries on your system
which will do evrything for you:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-spa \
                     poppler-utils graphviz graphviz-dev \
sudo apt-get install pandoc
# OR (for getting the newest version with all features)
# cd /tmp
# wget https://github.com/jgm/pandoc/releases/download/2.19.2/pandoc-2.19.2-1-amd64.deb
# dpkg -i pandoc-2.19.2-1-amd64.deb
```

Below are some explanation what the different

### Supporting \*.docx, \*.odt, \*.epub

In order to be able to load docx, odt and rtf files, you have to install pandoc.
Right now, the python pandoc library does not work with pandoc version > 3.0.0. It
is therefore recommended to install a version from here for your OS:

https://github.com/jgm/pandoc/releases/tag/2.19.2

### Image OCR Support

Pydoxtools can automatically analyze images as well, makin use of
[OCR](https://en.wikipedia.org/wiki/Optical_character_recognition).
In order to be able to use this, install tesseract on your system:

Under linux this looks like the following:

```bash
sudo apt-get update && sudo apt-get tesseract-ocr
# install tesseract languages 
# Display a list of all Tesseract language packs:
#   apt-cache search tesseract-ocr
# install all languages:
# sudo apt install tesseract-ocr-*
# install only german, french, english, spanish language packs
sudo apt install tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-spa
```

### pdf image rendering

For pdf rendering, Pydoxtools makes use of a library "poppler" which
needs to be installed on your system. Under linux, this looks like the following:

```bash
sudo apt-get install poppler-utils
```

### Graphviz

For visualizing the document logic, you need to install graphviz on your system.
Under linux, this looks like the following:

```bash
sudo apt-get install graphviz graphviz-dev
```

## Development

--> see [](DEVELOPMENT)

## License

This project is licensed under the terms of [MIT](LICENSE) license.

You can check the compatibility using the following tool in a venv environment in a production
setting:

    pip install pip-licenses
    pip-licenses | grep -Ev 'MIT License|BSD License|Apache Software License|Python Software Foundation License|Apache 2.0|MIT|Apache License 2.0|hnswlib|Pillow|new BSD|BSD'

### Dependencies

Here is a list of Libraries, that this project is based on:

[list](poetry.lock)

### Changelog

[changelog](CHANGELOG.md)