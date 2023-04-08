# pydoxtools

This library provides several functions in order to extract information from
all kinds of sources. 

It will analyze documents and do the following operations:

- extract tables
- other information (entities, addresses)
- provides a question & answering mechanism based on NLP
- extract specifications
- extract interfaces
- normalize the data
- translate the data
- clean the data

## Teaser

## Installation

### OCR

The ocr functionality relies on the Open Source Software
[tesseract](https://github.com/tesseract-ocr/tesseract) to extract
pdf files from images and image-based pdfs.

In order to install OCR functionality it's best to follow the instructions on their 
[webpage](https://tesseract-ocr.github.io/). In most linux distributions it
can be as simple as:

    sudo apt install tesseract-ocr

## License

This project is licensed under the terms of [MIT](./LICENSE) license.

You can check the compatibility using the following tool in a venv environment in a production
setting:

    pip install pip-licenses
    pip-licenses | grep -Ev 'MIT License|BSD License|Apache Software License|Python Software Foundation License|Apache 2.0|MIT|Apache License 2.0|hnswlib|Pillow|new BSD|BSD'
    
# libraries, that this project is based on:

