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

- Table extraction
- Vector Index Creation
- Document analysis and question-answering
- Entity, address identification and more
- List and keyword extraction
- Data normalization, translation, and cleaning

The library allows for the creation of complex extraction pipelines
for batch-processing of documents by defining them as a lazily-executed graph.

## Installation

While pydoxtools can already be installed through pip. Due to the
many updates coming in right now, it is right now recommended to use
the latest version from github as follows:

    pip install -U "pydoxtools[etl,inference] @ git+https://github.com/xyntopia/pydoxtools.git"

Pydoxtools can be also be installed through pip which will become the recommended
method once it becomes more stable:

    pip install -U pydoxtools[etl,inference]

For loading additional file formats (docx, odt, epub) and images, checkout
the additional > [Installation Options](#installation-options) <.

## Teaser

Experience a new level of convenience and efficiency in handling documents with Pydoxtools, and reimagine your data
extraction pipelines! 🎩✨📄.

    import pydoxtools as pdx

    # create a document from a file, string, bytestring, file-like object
    # or even an url:
    doc = pdx.Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf", 
        document_type=".pdf"
    )

easy extraction of a large number of information about your document to get a list
use `print(doc.x_funcs)`

    # extract tables from the pdf as a pandas dataframe:
    print(doc.tables_df)

some extraction operations need input when called:

    # ask a question about the document, using Q&A Models (questionas answered locally!):
    print(doc.answers(["how much power does it need?"])[0][0][0])

others need an API key installed, if it refers to an online service. 

    # ask a question about the document, using ChatGPT (we need the API key for ChatGPT!):
    # load the API key into an environment variable like this: 
    #   
    # OPENAI_API_KEY="sk ...."
    # 
    # Do **NOT** use the key in your code. This could potentially cost you a lot of money...
    print(doc.chat_answers(["What is the target group of this document?"])[0].content)
    print(doc.chat_answers(["Answer if a 5-year old would be able to follow these instructions?"])[0].content)

## Features

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

### PDF table extraction algorithms

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

### Image OCR support

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

## list of libraries, that this project is based on:

[list](poetry.lock)
