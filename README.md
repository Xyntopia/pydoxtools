# 🚀 pydoxtools 🚀

(*WIP*) [Documentation](https://xyntopia.github.io/pydoxtools)



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

## Teaser

Experience a new level of convenience and efficiency in handling documents with Pydoxtools, and reimagine your data extraction pipelines! 🎩✨📄.

    import pydoxtools as pdx

    # create a document from a file, string, bytestring, file-like object
    # or even an url:
    doc = Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf", 
        document_type=".pdf"
    )
    # extract the table as a pandas dataframe:
    print(doc.tables_df)
    # ask a question about the document, using Q&A Models (questionas answered locally!):
    print(doc.answers(["how much power does it need?"])[0][0][0])
    # ask a question about the document, using ChatGPT:
    print(doc.chat_answers(["who is the target group of this document?"])[0].content)
    print(doc.chat_answers(["Answer if a 5-year old would be able to follow these instructions?"])[0].content)


## Large pipelines

Pydoxtools main feature are large, composable and customizable pipelines. As a teaser
check out this pipeline for *.png images from the repository:

![Visualization of the pipeline for *.png images.](http://pydoxtools.xyntopia.com/images/document_logic_png.svg)

Pipelines can be mixed, partially overwritten and extended which gives you a lot of possibilities
to extend and adapt the functionality for your specific use-case. 

Find out more about it in the [documentation](http://pydoxtools.xyntopia.com/reference/#pydoxtools.document.Document)

## Installation

Pydoxtools can be installed through pip:

    pip install pydoxtools[etl, inference]

It should automatically download models & dependencies where required. In order to get the 
latest version install it like this:

    pip install git+https://github.com/xyntopia/pydoxtools'
    
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
