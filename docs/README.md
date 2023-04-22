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

When using pydoxtools with chatgpt, we need to make sure that you are using a 

    import pydoxtools as pdx

    # create a document from a file, string, bytestring, file-like object
    # or even an url:
    doc = Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf", 
        document_type=".pdf"
    )
    # extract the table as a pandas dataframe:
    print(doc.tables_df)
    print(doc.answers(["how much power does it need?"])[0][0][0])
    print(doc.chat_answers(["who is the target group of this document?"])[0].content)
    print(doc.chat_answers(["Answer if a 5-year old would be able to follow these instructions?"])[0].content)
    # ask a question about the document:
    

## CLI

TODO...

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