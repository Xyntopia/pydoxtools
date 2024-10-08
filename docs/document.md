# [pydoxtools.Document][]

::: pydoxtools.Document

## Text extraction attributes and functions

The [pydoxtools.Document][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

### DG

Alias for: 

\* document_graph-\>DG (output)

*name*
: `<Document>.x('DG') or <Document>.DG`

*return type*
: <class 'networkx.classes.digraph.DiGraph'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### a_d_ratio

Letter/digit ratio of the text

*name*
: `<Document>.x('a_d_ratio') or <Document>.a_d_ratio`

*return type*
: <class 'float'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### addresses

get addresses from text

*name*
: `<Document>.x('addresses') or <Document>.addresses`

*return type*
: list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### answers

Extract answers from the text using the Huggingface question answering pipeline

*name*
: `<Document>.x('answers') or <Document>.answers`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### chat_answers

Extract answers from the text using OpenAI Chat GPT and other models.

*name*
: `<Document>.x('chat_answers') or <Document>.chat_answers`

*return type*
: typing.Callable[[list[str], list[str] | str], list[str]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### clean_format

The format used to convert the document to a clean string for downstream processing tasks

*name*
: `<Document>.x('clean_format') or <Document>.clean_format`

*return type*
: typing.Any

*supports pipeline flows*
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### clean_spacy_text

Generate text to be used for spacy. Depending on the 'use_clean_text_for_spacy' option it will use page templates and replace complicated text structures such as tables for better text understanding.

*name*
: `<Document>.x('clean_spacy_text') or <Document>.clean_spacy_text`

*return type*
: <class 'str'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### clean_text

| pipe_type                                                                                                                                                                          | description                                                                                     |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
| \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/pdf, application/x-yaml, image, image/jpeg, image/png, image/tiff, text/html                                       | Alias for:                                                                                      |
|                                                                                                                                                                                    |                                                                                                 |
|                                                                                                                                                                                    | \* full_text-\>clean_text (output)                                                                |
| application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf | for some downstream tasks, it is better to have pure text, without any sructural elements in it |

*name*
: `<Document>.x('clean_text') or <Document>.clean_text`

*return type*
: <class 'str'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### convert_to

Generic pandoc converter for other document formats. TODO: better docs

*name*
: `<Document>.x('convert_to') or <Document>.convert_to`

*return type*
: typing.Callable

*supports pipeline flows*
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### coreferences

Resolve coreferences in the text

*name*
: `<Document>.x('coreferences') or <Document>.coreferences`

*return type*
: list[list[tuple[int, int]]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### data

| pipe_type                                                                                                                                                                                                                                         | description                                                         |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
| PIL.Image.Image                                                                                                                                                                                                                                   | Converts the image to a numpy array                                 |
| image, image/jpeg, image/png, image/tiff                                                                                                                                                                                                          | Converts the image to a numpy array for downstream processing tasks |
| application/x-yaml                                                                                                                                                                                                                                | Load yaml data from a string                                        |
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/html, text/markdown, text/rtf | The unprocessed data.                                               |

*name*
: `<Document>.x('data') or <Document>.data`

*return type*
: <class 'numpy.ndarray'\> | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### data_sel

select values by key from source data in Document

*name*
: `<Document>.x('data_sel') or <Document>.data_sel`

*return type*
: typing.Callable[..., dict]

*supports pipeline flows*
: <class 'dict'\>, application/x-yaml

### do

Alias for: 

\* document_objects-\>do (output)

*name*
: `<Document>.x('do') or <Document>.do`

*return type*
: dict[int, pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### document_graph

Builds a [networkx graph](https://networkx.org/documentation/stable/reference/classes/digraph.html) from the relations and coreferences

*name*
: `<Document>.x('document_graph') or <Document>.document_graph`

*return type*
: <class 'networkx.classes.digraph.DiGraph'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### document_objects

| pipe_type                                                                                                                                                                                                                                            | description                                                                   |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                                                                                           | extracts a list of document objects such as tables, text boxes, figures, etc. |
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/html, text/markdown, text/rtf | output a list of document elements which can be referenced by id              |

*name*
: `<Document>.x('document_objects') or <Document>.document_objects`

*return type*
: dict[int, pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### elements

| pipe_type                                                                                                                                                                          | description                                                                                 |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| application/pdf                                                                                                                                                                    | Loads a pdf file and returns a list of basic document elements such as lines, figures, etc. |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff                                                                                                                          | Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]              |
| \*, <class 'dict'\>, <class 'list'\>, application/x-yaml, text/html                                                                                                                   | extracts a list of document objects such as tables, text boxes, figures, etc.               |
| application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf | split a pandoc document into text elements.                                                 |

*name*
: `<Document>.x('elements') or <Document>.elements`

*return type*
: <class 'pandas.core.frame.DataFrame'\> | list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### embedded_meta

| pipe_type                                                                                                                                                                          | description                                  |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|
| application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf | Alias for:                                   |
|                                                                                                                                                                                    |                                              |
|                                                                                                                                                                                    | \* meta_pandoc-\>embedded_meta (output)        |
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                         | Alias for:                                   |
|                                                                                                                                                                                    |                                              |
|                                                                                                                                                                                    | \* meta_pdf-\>embedded_meta (output)           |
| \*, <class 'dict'\>, <class 'list'\>, application/x-yaml, text/html                                                                                                                   | represents the metadata embedded in the file |

*name*
: `<Document>.x('embedded_meta') or <Document>.embedded_meta`

*return type*
: <class 'dict'\> | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### embedding

Get a vector (embedding) for the entire text by taking the mean of the contextual embeddings of all tokens

*name*
: `<Document>.x('embedding') or <Document>.embedding`

*return type*
: list[float]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### entities

Extract entities from text

*name*
: `<Document>.x('entities') or <Document>.entities`

*return type*
: list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### file_meta

Some fast-to-calculate metadata information about a document

*name*
: `<Document>.x('file_meta') or <Document>.file_meta`

*return type*
: dict[str, typing.Any]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### final_urls

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('final_urls') or <Document>.final_urls`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### full_text

| pipe_type                                                                                                                                                                          | description                                                        |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
| text/html                                                                                                                                                                          | Alias for:                                                         |
|                                                                                                                                                                                    |                                                                    |
|                                                                                                                                                                                    | \* main_content-\>full_text (output)                                 |
| application/x-yaml                                                                                                                                                                 | Alias for:                                                         |
|                                                                                                                                                                                    |                                                                    |
|                                                                                                                                                                                    | \* raw_content-\>full_text (output)                                  |
| application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf | Converts the document to a string using pandoc                     |
| <class 'dict'\>                                                                                                                                                                     | Dump dict data to a yaml-like string                               |
| <class 'list'\>                                                                                                                                                                     | Dump list data to a yaml-like string                               |
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                         | Extracts the full text from the document by grouping text elements |
| \*                                                                                                                                                                                  | Full text as a string value                                        |

*name*
: `<Document>.x('full_text') or <Document>.full_text`

*return type*
: <class 'str'\> | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### goose_article

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('goose_article') or <Document>.goose_article`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### graphic_elements

Filters the document elements and only keeps the graphic elements

*name*
: `<Document>.x('graphic_elements') or <Document>.graphic_elements`

*return type*
: list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### headers

Extracts the headers from the document

*name*
: `<Document>.x('headers') or <Document>.headers`

*return type*
: list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### html_keywords

Extracts explicitly given keywords from the html document

*name*
: `<Document>.x('html_keywords') or <Document>.html_keywords`

*return type*
: set[str]

*supports pipeline flows*
: text/html

### html_keywords_str

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('html_keywords_str') or <Document>.html_keywords_str`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### image_elements

Filters the document elements and only keeps the image elements

*name*
: `<Document>.x('image_elements') or <Document>.image_elements`

*return type*
: list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### images

| pipe_type                                                 | description                                                                             |
|:----------------------------------------------------------|:----------------------------------------------------------------------------------------|
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff | Access images as a dictionary with page numbers as keys for downstream processing tasks |
| application/pdf                                           | Render a pdf into images which can be used for further downstream processing            |

*name*
: `<Document>.x('images') or <Document>.images`

*return type*
: dict[int, PIL.Image.Image]

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### items

Get the items of the dictionary

*name*
: `<Document>.x('items') or <Document>.items`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'dict'\>, application/x-yaml

### keys

Get the keys of the dictionary

*name*
: `<Document>.x('keys') or <Document>.keys`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'dict'\>, application/x-yaml

### keywords

| pipe_type                                                                                                                                                                                                                                                                                                             | description                                                                  |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| text/html                                                                                                                                                                                                                                                                                                             | Aggregates the keywords from the html document and found by other algorithms |
| \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/markdown, text/rtf | Alias for:                                                                   |
|                                                                                                                                                                                                                                                                                                                       |                                                                              |
|                                                                                                                                                                                                                                                                                                                       | \* textrank_keywords-\>keywords (output)                                       |

*name*
: `<Document>.x('keywords') or <Document>.keywords`

*return type*
: set[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### labeled_text_boxes

Classifies the text elements into addresses, emails, phone numbers, etc. if possible.

*name*
: `<Document>.x('labeled_text_boxes') or <Document>.labeled_text_boxes`

*return type*
: list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### language

| pipe_type                                                                                                                                                                                                                                 | description                                                                            |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/markdown, text/rtf | Detect language of a document, return 'unknown' in case of an error                    |
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                                                                                | Extracts the language of the document                                                  |
| text/html                                                                                                                                                                                                                                 | Extracts the main content from the html document, removing boilerplate and other noise |

*name*
: `<Document>.x('language') or <Document>.language`

*return type*
: <class 'str'\> | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### line_elements

Filters the document elements and only keeps the text elements

*name*
: `<Document>.x('line_elements') or <Document>.line_elements`

*return type*
: list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### lists

| pipe_type                                                                                                                                                                                                                                            | description                                    |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                                                                                           | Extracts lists from the document text elements |
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/html, text/markdown, text/rtf | Extracts the lists from the document           |

*name*
: `<Document>.x('lists') or <Document>.lists`

*return type*
: <class 'pandas.core.frame.DataFrame'\> | list[pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### main_content

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('main_content') or <Document>.main_content`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### main_content_clean_html

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('main_content_clean_html') or <Document>.main_content_clean_html`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### main_image

Extracts the main image from the html document

*name*
: `<Document>.x('main_image') or <Document>.main_image`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### meta

Metadata of the document

*name*
: `<Document>.x('meta') or <Document>.meta`

*return type*
: dict[str, typing.Any]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### meta_pandoc

meta information from pandoc document

*name*
: `<Document>.x('meta_pandoc') or <Document>.meta_pandoc`

*return type*
: typing.Any

*supports pipeline flows*
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### meta_pdf

| pipe_type                                                 | description                                                                                 |
|:----------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| application/pdf                                           | Loads a pdf file and returns a list of basic document elements such as lines, figures, etc. |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff | Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]              |

*name*
: `<Document>.x('meta_pdf') or <Document>.meta_pdf`

*return type*
: <class 'dict'\>

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### noun_chunks

Alias for: 

\* spacy_noun_chunks-\>noun_chunks (output)

*name*
: `<Document>.x('noun_chunks') or <Document>.noun_chunks`

*return type*
: typing.List[pydoxtools.document_base.TokenCollection]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_graph

Create a graph of similar nouns

*name*
: `<Document>.x('noun_graph') or <Document>.noun_graph`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_ids

Vectors for nouns and corresponding noun ids in order to find them in the spacy document

*name*
: `<Document>.x('noun_ids') or <Document>.noun_ids`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_index

Create an index for the nouns

*name*
: `<Document>.x('noun_index') or <Document>.noun_index`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_query

Create a query function for the nouns which can be used to do nearest-neighbor queries

*name*
: `<Document>.x('noun_query') or <Document>.noun_query`

*return type*
: typing.Callable[..., list[tuple]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_vecs

Vectors for nouns and corresponding noun ids in order to find them in the spacy document

*name*
: `<Document>.x('noun_vecs') or <Document>.noun_vecs`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_pages

| pipe_type                                                                                                                                                                                                                                            | description                                 |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/html, text/markdown, text/rtf | Number of pages in the document             |
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                                                                                           | Outputs the number of pages in the document |

*name*
: `<Document>.x('num_pages') or <Document>.num_pages`

*return type*
: <class 'int'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_sents

number of sentences

*name*
: `<Document>.x('num_sents') or <Document>.num_sents`

*return type*
: <class 'int'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_words

Number of words in the document

*name*
: `<Document>.x('num_words') or <Document>.num_words`

*return type*
: <class 'int'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### ocr_pdf_file

Extracts the text from the document using OCR. It does this by creating a pdf which is important in order to keep the positional information of the text elements.

*name*
: `<Document>.x('ocr_pdf_file') or <Document>.ocr_pdf_file`

*return type*
: typing.Any

*supports pipeline flows*
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### page_classifier

Classifies the pages into different types. This is useful for example for identifiying table of contents, certain chapters etc... . This works as a zero-shot classifier and the classes are not predefined. it can by called like this: 

Document('somefile.pdf').page_classifier(candidate_labels=['table_of_contents', 'credits', 'license'])

*name*
: `<Document>.x('page_classifier') or <Document>.page_classifier`

*return type*
: typing.Callable[[list[str]], dict]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_set

| pipe_type                                                                                                                                                                                                                                            | description                                                                                 |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/html, text/markdown, text/rtf | A constant value                                                                            |
| application/pdf                                                                                                                                                                                                                                      | Loads a pdf file and returns a list of basic document elements such as lines, figures, etc. |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff                                                                                                                                                                                            | Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]              |

*name*
: `<Document>.x('page_set') or <Document>.page_set`

*return type*
: set[int] | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates

Generates a text page while replacing certain elements of the page which can be specified as a list of ElementTypes. It also automatically replaces elements which don't have a textual representation with an identifier. This is often the case with images & figures for example. The Id of the placeholder refers to the index of the DocumentObject. So for example, if we encounter and Identifier:  {Table_22}, we would be able to find it using doc.document_objects[22] or doc.do[22].

*name*
: `<Document>.x('page_templates') or <Document>.page_templates`

*return type*
: typing.Callable[[list[str]], dict[int, str]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates_str

Outputs a nice text version of the documents with annotated document objects such as page numbers, tables, figures, etc.

*name*
: `<Document>.x('page_templates_str') or <Document>.page_templates_str`

*return type*
: <class 'str'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates_str_minimal

No documentation

*name*
: `<Document>.x('page_templates_str_minimal') or <Document>.page_templates_str_minimal`

*return type*
: <class 'str'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### pages_bbox

| pipe_type                                                 | description                                                                                 |
|:----------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| application/pdf                                           | Loads a pdf file and returns a list of basic document elements such as lines, figures, etc. |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff | Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]              |

*name*
: `<Document>.x('pages_bbox') or <Document>.pages_bbox`

*return type*
: <class 'numpy.ndarray'\>

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### pandoc_document

Loads the document using the pandoc project [https://pandoc.org/](https://pandoc.org/) into a pydoxtools list of [][pydoxtools.document_base.DocumentElement]

*name*
: `<Document>.x('pandoc_document') or <Document>.pandoc_document`

*return type*
: Pandoc(Meta, [Block])

*supports pipeline flows*
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### pdf_links

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('pdf_links') or <Document>.pdf_links`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### pil_image

| pipe_type                                | description                                                             |
|:-----------------------------------------|:------------------------------------------------------------------------|
| PIL.Image.Image                          | Alias for:                                                              |
|                                          |                                                                         |
|                                          | \* _fobj-\>pil_image (output)                                             |
| image, image/jpeg, image/png, image/tiff | Converts the image to a PIL-style image for downstream processing tasks |

*name*
: `<Document>.x('pil_image') or <Document>.pil_image`

*return type*
: <class 'PIL.Image.Image'\> | typing.Any

*supports pipeline flows*
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### schemadata

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('schemadata') or <Document>.schemadata`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### sections

Extracts the sections from the document by grouping text elements

*name*
: `<Document>.x('sections') or <Document>.sections`

*return type*
: typing.Any

*supports pipeline flows*
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### segment_query

Create a query function for the text segments which can be used to do nearest-neighbor queries

*name*
: `<Document>.x('segment_query') or <Document>.segment_query`

*return type*
: typing.Callable[..., list[tuple]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### semantic_relations

Extract relations from text for building a knowledge graph

*name*
: `<Document>.x('semantic_relations') or <Document>.semantic_relations`

*return type*
: <class 'pandas.core.frame.DataFrame'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_graph

Create a graph of similar sentences

*name*
: `<Document>.x('sent_graph') or <Document>.sent_graph`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_ids

Vectors for sentences & sentence_ids

*name*
: `<Document>.x('sent_ids') or <Document>.sent_ids`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_index

Create an index for the sentences

*name*
: `<Document>.x('sent_index') or <Document>.sent_index`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_query

Create a query function for the sentences which can be used to do nearest-neighbor queries

*name*
: `<Document>.x('sent_query') or <Document>.sent_query`

*return type*
: typing.Callable[..., list[tuple]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_vecs

Vectors for sentences & sentence_ids

*name*
: `<Document>.x('sent_vecs') or <Document>.sent_vecs`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sents

Alias for: 

\* spacy_sents-\>sents (output)

*name*
: `<Document>.x('sents') or <Document>.sents`

*return type*
: list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### short_title

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('short_title') or <Document>.short_title`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### side_titles

Extracts the titles from the document by detecting unusual font styles

*name*
: `<Document>.x('side_titles') or <Document>.side_titles`

*return type*
: typing.Any

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### slow_summary

Summarize the text using the Huggingface summarization pipeline

*name*
: `<Document>.x('slow_summary') or <Document>.slow_summary`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_doc

Spacy Document and Language Model for this document

*name*
: `<Document>.x('spacy_doc') or <Document>.spacy_doc`

*return type*
: <class 'spacy.tokens.doc.Doc'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_embeddings

Embeddings calculated by a spacy transformer

*name*
: `<Document>.x('spacy_embeddings') or <Document>.spacy_embeddings`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_nlp

Spacy Document and Language Model for this document

*name*
: `<Document>.x('spacy_nlp') or <Document>.spacy_nlp`

*return type*
: <class 'spacy.language.Language'\>

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_noun_chunks

exracts nounchunks from spacy. Will not be cached because it is allin the spacy doc already

*name*
: `<Document>.x('spacy_noun_chunks') or <Document>.spacy_noun_chunks`

*return type*
: typing.List[pydoxtools.document_base.TokenCollection]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_sents

List of sentences by spacy nlp framework

*name*
: `<Document>.x('spacy_sents') or <Document>.spacy_sents`

*return type*
: list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_vectorizer

Create a vectorizer function from spacy library.

*name*
: `<Document>.x('spacy_vectorizer') or <Document>.spacy_vectorizer`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_vectors

Vectors for all tokens calculated by spacy

*name*
: `<Document>.x('spacy_vectors') or <Document>.spacy_vectors`

*return type*
: torch.Tensor | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### summary

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('summary') or <Document>.summary`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### table_areas

Areas of all detected tables

*name*
: `<Document>.x('table_areas') or <Document>.table_areas`

*return type*
: list[numpy.ndarray]

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_box_levels

| pipe_type                                                 | description                                                                                                                                                                                                   |
|:----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| application/pdf                                           | Detects table candidates from the document elements                                                                                                                                                           |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff | Extracts the table candidates from the document. As this is an image, we need to use a different method than for pdfs. Right now this relies on neural networks. TODO: add adtitional pure text-based method. |

*name*
: `<Document>.x('table_box_levels') or <Document>.table_box_levels`

*return type*
: typing.Any

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_candidates

| pipe_type                                                 | description                                                                                                                                                                                                   |
|:----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| application/pdf                                           | Detects table candidates from the document elements                                                                                                                                                           |
| PIL.Image.Image, image, image/jpeg, image/png, image/tiff | Extracts the table candidates from the document. As this is an image, we need to use a different method than for pdfs. Right now this relies on neural networks. TODO: add adtitional pure text-based method. |

*name*
: `<Document>.x('table_candidates') or <Document>.table_candidates`

*return type*
: typing.Any

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_context

Outputs a dictionary with the context of each table in the document

*name*
: `<Document>.x('table_context') or <Document>.table_context`

*return type*
: dict[int, str]

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_df0

Filter valid tables from table candidates by looking if meaningful values can be extracted

*name*
: `<Document>.x('table_df0') or <Document>.table_df0`

*return type*
: list[pandas.core.frame.DataFrame]

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### tables

Extracts the tables from the document as a document element

*name*
: `<Document>.x('tables') or <Document>.tables`

*return type*
: dict[int, pydoxtools.document_base.DocumentElement]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tables_df

| pipe_type                                                                                                                                                                                                                                 | description                                                                            |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                                                                                | Dataframes of all tables                                                               |
| text/html                                                                                                                                                                                                                                 | Extracts the main content from the html document, removing boilerplate and other noise |
| \*, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, mediawiki, pandoc, text/markdown, text/rtf | No documentation                                                                       |

*name*
: `<Document>.x('tables_df') or <Document>.tables_df`

*return type*
: list[pandas.core.frame.DataFrame] | typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tables_dict

List of Table

*name*
: `<Document>.x('tables_dict') or <Document>.tables_dict`

*return type*
: list[dict[int, dict[int, str]]]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_box_elements

| pipe_type                                                                                                                                                                             | description                                                                                                                                         |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| <class 'dict'\>, application/x-yaml                                                                                                                                                    | Create a dataframe from a dictionary. TODO: this is not working correctly, it should create a list of [][pydoxtools.document_base.DocumentELements] |
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff                                                                                                            | Extracts a dataframe of text boxes from the document by grouping text elements                                                                      |
| text/html                                                                                                                                                                             | Extracts the text boxes from the html document                                                                                                      |
| <class 'list'\>                                                                                                                                                                        | No documentation                                                                                                                                    |
| \*, application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf | Text boxes extracted as a pandas Dataframe with some additional metadata                                                                            |

*name*
: `<Document>.x('text_box_elements') or <Document>.text_box_elements`

*return type*
: list[pydoxtools.document_base.DocumentElement] | list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_ids

Get the a list of ids for individual text segments

*name*
: `<Document>.x('text_segment_ids') or <Document>.text_segment_ids`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_index

Create an index for the text segments

*name*
: `<Document>.x('text_segment_index') or <Document>.text_segment_index`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_vec_res

Calculate the embeddings for each text segment

*name*
: `<Document>.x('text_segment_vec_res') or <Document>.text_segment_vec_res`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_vecs

Get the embeddings for individual text segments

*name*
: `<Document>.x('text_segment_vecs') or <Document>.text_segment_vecs`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segments

Split the text into segments

*name*
: `<Document>.x('text_segments') or <Document>.text_segments`

*return type*
: list[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### textrank_keywords

Extract keywords from the graph of similar nouns

*name*
: `<Document>.x('textrank_keywords') or <Document>.textrank_keywords`

*return type*
: set[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### textrank_sents

Extract the most important sentences from the graph of similar sentences

*name*
: `<Document>.x('textrank_sents') or <Document>.textrank_sents`

*return type*
: set[str]

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### title

Extracts the main content from the html document, removing boilerplate and other noise

*name*
: `<Document>.x('title') or <Document>.title`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### titles

| pipe_type                                                                  | description                                                            |
|:---------------------------------------------------------------------------|:-----------------------------------------------------------------------|
| PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff | Extracts the titles from the document by detecting unusual font styles |
| text/html                                                                  | Extracts the titles from the html document                             |

*name*
: `<Document>.x('titles') or <Document>.titles`

*return type*
: tuple[str, str] | typing.Any

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff, text/html

### tok_embeddings

Get the tokenized text

*name*
: `<Document>.x('tok_embeddings') or <Document>.tok_embeddings`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tokens

Get the tokenized text

*name*
: `<Document>.x('tokens') or <Document>.tokens`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### url

| pipe_type                                                                                                                                                                                                                                                                                                             | description                                                                            |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| text/html                                                                                                                                                                                                                                                                                                             | Extracts the main content from the html document, removing boilerplate and other noise |
| \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/markdown, text/rtf | Url of this document                                                                   |

*name*
: `<Document>.x('url') or <Document>.url`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### urls

Extracts the urls from the html document

*name*
: `<Document>.x('urls') or <Document>.urls`

*return type*
: typing.Any

*supports pipeline flows*
: text/html

### valid_tables

Filter valid tables from table candidates by looking if meaningful values can be extracted

*name*
: `<Document>.x('valid_tables') or <Document>.valid_tables`

*return type*
: typing.Any

*supports pipeline flows*
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### values

Get the values of the dictionary

*name*
: `<Document>.x('values') or <Document>.values`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'dict'\>, application/x-yaml

### vec_res

Calculate context-based vectors (embeddings) for the entire text

*name*
: `<Document>.x('vec_res') or <Document>.vec_res`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vector

Embeddings from spacy

*name*
: `<Document>.x('vector') or <Document>.vector`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vectorizer

Get the vectorizer function used for this document for an arbitrary text

*name*
: `<Document>.x('vectorizer') or <Document>.vectorizer`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### Configuration parameters

| name                      | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | default_values                         |
|:--------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|
| chat_model_id             | In order to use openai-chatgpt, you can use 'gpt-3.5-turbo' or 'gpt-4'.Additionally, we support models used by gpt4all library whichcan be run locally and most are available for commercial purposes. Currently available models are: ['wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0', 'ggml-model-gpt4all-falcon-q4_0', 'ous-hermes-13b.ggmlv3.q4_0', 'GPT4All-13B-snoozy.ggmlv3.q4_0', 'orca-mini-7b.ggmlv3.q4_0', 'orca-mini-3b.ggmlv3.q4_0', 'orca-mini-13b.ggmlv3.q4_0', 'wizardLM-13B-Uncensored.ggmlv3.q4_0', 'ggml-replit-code-v1-3', 'ggml-all-MiniLM-L6-v2-f16', 'starcoderbase-3b-ggml', 'starcoderbase-7b-ggml', 'llama-2-7b-chat.ggmlv3.q4_0'] | gpt-3.5-turbo                          |
| coreference_method        | can be 'fast' or 'accurate'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | fast                                   |
| full_text_format          | The format used to convert the document to a string                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | markdown                               |
| graph_debug_context_size  | can be 'fast' or 'accurate'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 0                                      |
| image_dpi                 | The dpi when rendering the document. The standard image generation resolution is set to 216 dpi for pdfs as we want to have sufficient DPI for downstram OCR tasks (e.g. table extraction)                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 216                                    |
| max_size_text_segment     | controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 512                                    |
| max_text_segment_num      | controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 100                                    |
| min_size_text_segment     | controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 256                                    |
| ocr_lang                  | Configuration for the ocr extractor. We can turn it on/off and specify the language used for OCR.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | auto                                   |
| ocr_on                    | Configuration for the ocr extractor. We can turn it on/off and specify the language used for OCR.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | True                                   |
| qam_model_id              | Configuration for values:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | deepset/minilm-uncased-squad2          |
|                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|                           | \* qam_model_id = deepset/minilm-uncased-squad2 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                        |
| spacy_model               | we can also explicitly specify the spacy model we want to use.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | auto                                   |
| spacy_model_size          | the model size which is used for spacy text analysis. Can be:  sm,md,lg,trf.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | md                                     |
| summarizer_max_text_len   | Configuration for values:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 200                                    |
|                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|                           | \* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                        |
|                           | \* summarizer_token_overlap = 50 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
|                           | \* summarizer_max_text_len = 200 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
| summarizer_model          | Configuration for values:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | sshleifer/distilbart-cnn-12-6          |
|                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|                           | \* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                        |
|                           | \* summarizer_token_overlap = 50 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
|                           | \* summarizer_max_text_len = 200 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
| summarizer_token_overlap  | Configuration for values:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 50                                     |
|                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|                           | \* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                        |
|                           | \* summarizer_token_overlap = 50 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
|                           | \* summarizer_max_text_len = 200 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                        |
| text_segment_overlap      | controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 0.3                                    |
| top_k_text_rank_keywords  | Configuration for values:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 5                                      |
|                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|                           | \* top_k_text_rank_keywords = 5 (default)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                        |
| top_k_text_rank_sentences | controls the number of most important sentences that are extracted from the text.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 5                                      |
| use_clean_text_for_spacy  | Whether pydoxtools cleans up the text before using spacy on it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | True                                   |
| vectorizer_model          | Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.                                                                                                                                                                                                                                                                                                                                                               | sentence-transformers/all-MiniLM-L6-v2 |
| vectorizer_only_tokenizer | Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.                                                                                                                                                                                                                                                                                                                                                               | False                                  |
| vectorizer_overlap_ratio  | Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.                                                                                                                                                                                                                                                                                                                                                               | 0.1                                    |