# pydoxtools

Pydoxtools is a library that provides a sophisticated interface for reading and
writing documents, designed to work with AI models such as GPT, Alpaca, and
Huggingface. It offers functionalities such as:

- Table extraction
- Document analysis and question-answering
- Task relation creation
- Entity, address identification and more
- List and keyword extraction
- Data normalization, translation, and cleaning

The library allows for the creation of complex extraction pipelines
for batch-processing of documents by defining them as a lazily-executed graph.

## Teaser

    import pydoxtools as pdx

    # create a document from a file, string, bytestring, file-like object
    # or even an url:
    pdx.document("")

## Installation

    pip install pydoxtools[etl, inference]

    # TODO: explain some dependencies (especially pytorch) 

## Examples

- analyze documents using any sort of model from huggingface...
- analyze documents using a custom model
- download a pdf from URL
- generate document keywords
- extract tables
- download document from URL "manually" and then feed to document
- extract addresses
- extract addresses and use this information for the qam

## Development

--> see [](DEVELOPMENT.md)

## License

This project is licensed under the terms of [MIT](./LICENSE) license.

You can check the compatibility using the following tool in a venv environment in a production
setting:

    pip install pip-licenses
    pip-licenses | grep -Ev 'MIT License|BSD License|Apache Software License|Python Software Foundation License|Apache 2.0|MIT|Apache License 2.0|hnswlib|Pillow|new BSD|BSD'

## list of libraries, that this project is based on:

[list](poetry.lock)
