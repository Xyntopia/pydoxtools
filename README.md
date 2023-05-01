🚀 pydoxtools (Python Library) 🚀
================================================================================

![Python](https://img.shields.io/pypi/pyversions/pydoxtools.svg)
[![PyPI version](https://img.shields.io/pypi/v/pydoxtools.svg)](https://pypi.python.org/pypi/pydoxtools)
[![Mkdocs](https://img.shields.io/badge/doc-mkdocs-845ed7.svg)](https://pydoxtools.xyntopia.com)
[![GitHub discussions](https://img.shields.io/badge/discuss-online-845ef7)](https://github.com/xyntopia/pydoxtools/discussions)
[![GitHub stars](https://img.shields.io/github/stars/xyntopia/pydoxtools)](https://github.com/xyntopia/pydoxtools/stargazers)

***

(*WIP*) [Documentation](https://pydoxtools.xyntopia.com)

Pydoxtools is a library that provides a sophisticated interface for reading and
writing documents, designed to work with AI models such as GPT, Alpaca, and
Huggingface. It offers functionalities such as:

- Pipeline management on a new level
- Integration with AI (LLMs and more) models
- (PDF) table extraction without configuration!
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

For loading additional file formats (docx, odt, epub) and images, check out
the additional > [Installation Options](#installation-options) <.

## Teaser

Experience a new level of convenience and efficiency in handling
documents with Pydoxtools, and reimagine your data extraction pipelines! 🎩✨📄.

In this teaser, we'll demonstrate how to create a document, extract
tables, and ask questions using AI models:

```python
import pydoxtools as pdx

# Create a document from various sources: file, string, bytestring, file-like object, or URL
doc = pdx.Document(
    "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf",
    document_type=".pdf"
)

# List available extraction functions
print(doc.x_funcs)

# Extract tables from the PDF as a pandas DataFrame
print(doc.tables_df)

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

## Some Features in more Detail

### Large pipelines

Pydoxtools main feature is the ability to mix LLMs and other
AI models in large, composable and customizable pipelines.
As a teaser, check out this pipeline for *.png images from the repository including
OCR, keyword extraction, vectorization and more. In this pipeline:

- Every node in an ellipse can be called as an attribute of the document-analysis pipeline.
- Every execution-path is lazily executed throughout the entire graph.
- Every node is cached by default (can be turned off).
- Every piece of this pipeline can be replaced by a customized version.

<img src="http://pydoxtools.xyntopia.com/images/document_logic_png.svg" width="500">

Pipelines can be mixed, partially overwritten and extended which gives you a lot of possibilities
to extend and adapt the functionality for your specific use-case.

Find out more about it in the [documentation](http://pydoxtools.xyntopia.com/reference/#pydoxtools.document.Document)

#### Pipeline Configuration

Pipelines can be configured. For example the local model used for
question answering can be selected like this:

    doc = Document(fobj="./data/PFR-PR23_BAT-110__V1.00_.pdf"))
            .config(qam_model_id='bert-large-uncased-whole-word-masking-finetuned-squad')

where "qam_model_id" can be any model from huggingface for question answering.

    TODO: document how to configure a pipeline

### PDF Table Extraction Algorithms

The library features its own sophisticated Table extraction algorithm which is benchmarked
against a large pdf table dataset. In contrast to most other table extraction frameworks
out there it does not require:

- extensive configuration
- no expensive deep neural networks which need a GPU

This makes it possible to run analysis on PDF files with pydoxtools on CPU with
very limited resources!

### TODO: Describe more of the features here...

## Use Cases

- analyze documents using any model from huggingface...
- analyze documents using a custom model
- download a pdf from URL
- generate document keywords
- extract tables
- download document from URL "manually" and then feed to document
- extract addresses
- extract addresses and use this information for the qam
- ingest documents into a vector db

## Installation Options

### Supporting *.docx, *.odt, *.epub

In order to be able to load docx, odt and rtf files, you have to install pandoc.
Right now, the python pandoc library does not work with pandoc version > 3.0.0. It
is therefore recommended to install a version from here for your OS:

https://github.com/jgm/pandoc/releases/tag/2.19.2

### Image OCR Support

Pydoxtools can automatically analyze images as well, makin use of
[OCR](https://en.wikipedia.org/wiki/Optical_character_recognition).
In order to be able to use this, install tesseract on your system:

Under linux this looks like the following:

    apt-get update && tesseract-ocr
    # install tesseract languages 
    # Display a list of all Tesseract language packs:
    #   apt-cache search tesseract-ocr
    # install all languages:
    # sudo apt install tesseract-ocr-*
    # install only german, french, english, spanish language packs
    # sudo apt install tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-spa

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
