# [pydoxtools.Document][]

::: pydoxtools.Document

## Text extraction attributes and functions

The [pydoxtools.Document][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

### data
            
The unprocessed data.

Can be called using:

    <Document>.x('data')
    # or
    <Document>.data

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_set
            
A constant value

Can be called using:

    <Document>.x('page_set')
    # or
    <Document>.page_set

return type
: set[int] | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### full_text
            
Full text as a string value

Can be called using:

    <Document>.x('full_text')
    # or
    <Document>.full_text

return type
: <class 'str'\> | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### clean_text
            
Alias for: 

* full_text->clean_text (output)

Can be called using:

    <Document>.x('clean_text')
    # or
    <Document>.clean_text

return type
: <class 'str'\> | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tables_df
            
No documentation

Can be called using:

    <Document>.x('tables_df')
    # or
    <Document>.tables_df

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tables_dict
            
List of Table

Can be called using:

    <Document>.x('tables_dict')
    # or
    <Document>.tables_dict

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tables
            
Extracts the tables from the document as a dataframe

Can be called using:

    <Document>.x('tables')
    # or
    <Document>.tables

return type
: list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### elements
            
extracts a list of document objects such as tables, text boxes, figures, etc.

Can be called using:

    <Document>.x('elements')
    # or
    <Document>.elements

return type
: <class 'pandas.core.frame.DataFrame'\> | list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### document_objects
            
Alias for: 

* elements->document_objects (output)

Can be called using:

    <Document>.x('document_objects')
    # or
    <Document>.document_objects

return type
: list[pydoxtools.document_base.DocumentElement] | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_box_elements
            
Text boxes extracted as a pandas Dataframe with some additional metadata

Can be called using:

    <Document>.x('text_box_elements')
    # or
    <Document>.text_box_elements

return type
: pandas.core.frame.DataFrame | None | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### headers
            
Extracts the headers from the document

Can be called using:

    <Document>.x('headers')
    # or
    <Document>.headers

return type
: list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### lists
            
Extracts the lists from the document

Can be called using:

    <Document>.x('lists')
    # or
    <Document>.lists

return type
: <class 'pandas.core.frame.DataFrame'\> | list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### line_elements
            
Filters the document elements and only keeps the text elements

Can be called using:

    <Document>.x('line_elements')
    # or
    <Document>.line_elements

return type
: list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### graphic_elements
            
Filters the document elements and only keeps the graphic elements

Can be called using:

    <Document>.x('graphic_elements')
    # or
    <Document>.graphic_elements

return type
: list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### image_elements
            
Filters the document elements and only keeps the image elements

Can be called using:

    <Document>.x('image_elements')
    # or
    <Document>.image_elements

return type
: list[pydoxtools.document_base.DocumentElement]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates
            
generates a text page with table & figure hints

Can be called using:

    <Document>.x('page_templates')
    # or
    <Document>.page_templates

return type
: typing.Callable[[list[str]], dict[int, str]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates_str
            
Outputs a nice text version of the documents with annotated document objects such as page numbers, tables, figures, etc.

Can be called using:

    <Document>.x('page_templates_str')
    # or
    <Document>.page_templates_str

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_templates_str_minimal
            
No documentation

Can be called using:

    <Document>.x('page_templates_str_minimal')
    # or
    <Document>.page_templates_str_minimal

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### addresses
            
Classifies the text elements into addresses, emails, phone numbers, etc. if possible.

Can be called using:

    <Document>.x('addresses')
    # or
    <Document>.addresses

return type
: list[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### page_classifier
            
Classifies the pages into different types. This is useful for example for identifiying table of contents, certain chapters etc... . This works as a zero-shot classifier and the classes are not predefined. it can by called like this: 

Document('somefile.pdf').page_classifier(candidate_labels=['table_of_contents', 'credits', 'license'])

Can be called using:

    <Document>.x('page_classifier')
    # or
    <Document>.page_classifier

return type
: typing.Callable[[list[str]], dict]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### embedded_meta
            
represents the metadata embedded in the file

Can be called using:

    <Document>.x('embedded_meta')
    # or
    <Document>.embedded_meta

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### meta
            
Metadata of the document

Can be called using:

    <Document>.x('meta')
    # or
    <Document>.meta

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_pages
            
Number of pages in the document

Can be called using:

    <Document>.x('num_pages')
    # or
    <Document>.num_pages

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_words
            
Number of words in the document

Can be called using:

    <Document>.x('num_words')
    # or
    <Document>.num_words

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### num_sents
            
number of sentences

Can be called using:

    <Document>.x('num_sents')
    # or
    <Document>.num_sents

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### a_d_ratio
            
Letter/digit ratio of the text

Can be called using:

    <Document>.x('a_d_ratio')
    # or
    <Document>.a_d_ratio

return type
: <class 'float'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### language
            
Detect language of a document, return 'unknown' in case of an error

Can be called using:

    <Document>.x('language')
    # or
    <Document>.language

return type
: <class 'str'\> | typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### file_meta
            
Some fast-to-calculate metadata information about a document

Can be called using:

    <Document>.x('file_meta')
    # or
    <Document>.file_meta

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_model_size
            
Configuration for values:

* spacy_model_size = md (default)
* spacy_model = auto (default)

Can be called using:

    <Document>.x('spacy_model_size')
    # or
    <Document>.spacy_model_size

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_model
            
Configuration for values:

* spacy_model_size = md (default)
* spacy_model = auto (default)

Can be called using:

    <Document>.x('spacy_model')
    # or
    <Document>.spacy_model

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_doc
            
Spacy Document and Language Model for this document

Can be called using:

    <Document>.x('spacy_doc')
    # or
    <Document>.spacy_doc

return type
: spacy.language.Language | spacy.tokens.doc.Doc

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_nlp
            
Spacy Document and Language Model for this document

Can be called using:

    <Document>.x('spacy_nlp')
    # or
    <Document>.spacy_nlp

return type
: spacy.language.Language | spacy.tokens.doc.Doc

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_vectors
            
Vectors for all tokens calculated by spacy

Can be called using:

    <Document>.x('spacy_vectors')
    # or
    <Document>.spacy_vectors

return type
: typing.Union[torch.Tensor, typing.Any]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_embeddings
            
Embeddings calculated by a spacy transformer

Can be called using:

    <Document>.x('spacy_embeddings')
    # or
    <Document>.spacy_embeddings

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_sents
            
List of sentences by spacy nlp framework

Can be called using:

    <Document>.x('spacy_sents')
    # or
    <Document>.spacy_sents

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_noun_chunks
            
exracts nounchunks from spacy. Will not be cached because it is allin the spacy doc already

Can be called using:

    <Document>.x('spacy_noun_chunks')
    # or
    <Document>.spacy_noun_chunks

return type
: typing.List[pydoxtools.document_base.TokenCollection]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### entities
            
Extract entities from text

Can be called using:

    <Document>.x('entities')
    # or
    <Document>.entities

return type
: list[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### url
            
Url of this document

Can be called using:

    <Document>.x('url')
    # or
    <Document>.url

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### semantic_relations
            
Extract relations from text for building a knowledge graph

Can be called using:

    <Document>.x('semantic_relations')
    # or
    <Document>.semantic_relations

return type
: <class 'pandas.core.frame.DataFrame'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### coreference_method
            
can be 'fast' or 'accurate'

Can be called using:

    <Document>.x('coreference_method')
    # or
    <Document>.coreference_method

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### graph_debug_context_size
            
can be 'fast' or 'accurate'

Can be called using:

    <Document>.x('graph_debug_context_size')
    # or
    <Document>.graph_debug_context_size

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### coreferences
            
Resolve coreferences in the text

Can be called using:

    <Document>.x('coreferences')
    # or
    <Document>.coreferences

return type
: list[list[tuple[int, int]]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### document_graph
            
Builds a [networkx graph](https://networkx.org/documentation/stable/reference/classes/digraph.html) from the relations and coreferences

Can be called using:

    <Document>.x('document_graph')
    # or
    <Document>.document_graph

return type
: <class 'networkx.classes.digraph.DiGraph'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### DG
            
Alias for: 

* document_graph->DG (output)

Can be called using:

    <Document>.x('DG')
    # or
    <Document>.DG

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sents
            
Alias for: 

* spacy_sents->sents (output)

Can be called using:

    <Document>.x('sents')
    # or
    <Document>.sents

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_chunks
            
Alias for: 

* spacy_noun_chunks->noun_chunks (output)

Can be called using:

    <Document>.x('noun_chunks')
    # or
    <Document>.noun_chunks

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vector
            
Embeddings from spacy

Can be called using:

    <Document>.x('vector')
    # or
    <Document>.vector

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_vecs
            
Vectors for sentences & sentence_ids

Can be called using:

    <Document>.x('sent_vecs')
    # or
    <Document>.sent_vecs

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_ids
            
Vectors for sentences & sentence_ids

Can be called using:

    <Document>.x('sent_ids')
    # or
    <Document>.sent_ids

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_vecs
            
Vectors for nouns and corresponding noun ids in order to find them in the spacy document

Can be called using:

    <Document>.x('noun_vecs')
    # or
    <Document>.noun_vecs

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_ids
            
Vectors for nouns and corresponding noun ids in order to find them in the spacy document

Can be called using:

    <Document>.x('noun_ids')
    # or
    <Document>.noun_ids

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vectorizer_model
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_model')
    # or
    <Document>.vectorizer_model

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vectorizer_only_tokenizer
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_only_tokenizer')
    # or
    <Document>.vectorizer_only_tokenizer

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vectorizer_overlap_ratio
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_overlap_ratio')
    # or
    <Document>.vectorizer_overlap_ratio

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vectorizer
            
Get the vectorizer function used for this document for an arbitrary text

Can be called using:

    <Document>.x('vectorizer')
    # or
    <Document>.vectorizer

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### vec_res
            
Calculate context-based vectors (embeddings) for the entire text

Can be called using:

    <Document>.x('vec_res')
    # or
    <Document>.vec_res

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tok_embeddings
            
Get the tokenized text

Can be called using:

    <Document>.x('tok_embeddings')
    # or
    <Document>.tok_embeddings

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### tokens
            
Get the tokenized text

Can be called using:

    <Document>.x('tokens')
    # or
    <Document>.tokens

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### embedding
            
Get a vector (embedding) for the entire text by taking the mean of the contextual embeddings of all tokens

Can be called using:

    <Document>.x('embedding')
    # or
    <Document>.embedding

return type
: list[float]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### min_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('min_size_text_segment')
    # or
    <Document>.min_size_text_segment

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### max_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('max_size_text_segment')
    # or
    <Document>.max_size_text_segment

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_overlap
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('text_segment_overlap')
    # or
    <Document>.text_segment_overlap

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### max_text_segment_num
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('max_text_segment_num')
    # or
    <Document>.max_text_segment_num

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segments
            
Split the text into segments

Can be called using:

    <Document>.x('text_segments')
    # or
    <Document>.text_segments

return type
: list[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_vec_res
            
Calculate the embeddings for each text segment

Can be called using:

    <Document>.x('text_segment_vec_res')
    # or
    <Document>.text_segment_vec_res

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_vecs
            
Get the embeddings for individual text segments

Can be called using:

    <Document>.x('text_segment_vecs')
    # or
    <Document>.text_segment_vecs

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_ids
            
Get the a list of ids for individual text segments

Can be called using:

    <Document>.x('text_segment_ids')
    # or
    <Document>.text_segment_ids

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### text_segment_index
            
Create an index for the text segments

Can be called using:

    <Document>.x('text_segment_index')
    # or
    <Document>.text_segment_index

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### segment_query
            
Create a query function for the text segments which can be used to do nearest-neighbor queries

Can be called using:

    <Document>.x('segment_query')
    # or
    <Document>.segment_query

return type
: typing.Callable[..., list[tuple]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_index
            
Create an index for the nouns

Can be called using:

    <Document>.x('noun_index')
    # or
    <Document>.noun_index

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### spacy_vectorizer
            
Create a vectorizer function from spacy library.

Can be called using:

    <Document>.x('spacy_vectorizer')
    # or
    <Document>.spacy_vectorizer

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_query
            
Create a query function for the nouns which can be used to do nearest-neighbor queries

Can be called using:

    <Document>.x('noun_query')
    # or
    <Document>.noun_query

return type
: typing.Callable[..., list[tuple]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### noun_graph
            
Create a graph of similar nouns

Can be called using:

    <Document>.x('noun_graph')
    # or
    <Document>.noun_graph

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### top_k_text_rank_keywords
            
Configuration for values:

* top_k_text_rank_keywords = 5 (default)

Can be called using:

    <Document>.x('top_k_text_rank_keywords')
    # or
    <Document>.top_k_text_rank_keywords

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### textrank_keywords
            
Extract keywords from the graph of similar nouns

Can be called using:

    <Document>.x('textrank_keywords')
    # or
    <Document>.textrank_keywords

return type
: set[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### keywords
            
Alias for: 

* textrank_keywords->keywords (output)

Can be called using:

    <Document>.x('keywords')
    # or
    <Document>.keywords

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_index
            
Create an index for the sentences

Can be called using:

    <Document>.x('sent_index')
    # or
    <Document>.sent_index

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_query
            
Create a query function for the sentences which can be used to do nearest-neighbor queries

Can be called using:

    <Document>.x('sent_query')
    # or
    <Document>.sent_query

return type
: typing.Callable[..., list[tuple]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### sent_graph
            
Create a graph of similar sentences

Can be called using:

    <Document>.x('sent_graph')
    # or
    <Document>.sent_graph

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### top_k_text_rank_sentences
            
controls the number of most important sentences that are extracted from the text.

Can be called using:

    <Document>.x('top_k_text_rank_sentences')
    # or
    <Document>.top_k_text_rank_sentences

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### textrank_sents
            
Extract the most important sentences from the graph of similar sentences

Can be called using:

    <Document>.x('textrank_sents')
    # or
    <Document>.textrank_sents

return type
: set[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### summarizer_model
            
Configuration for values:

* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)
* summarizer_token_overlap = 50 (default)
* summarizer_max_text_len = 200 (default)

Can be called using:

    <Document>.x('summarizer_model')
    # or
    <Document>.summarizer_model

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### summarizer_token_overlap
            
Configuration for values:

* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)
* summarizer_token_overlap = 50 (default)
* summarizer_max_text_len = 200 (default)

Can be called using:

    <Document>.x('summarizer_token_overlap')
    # or
    <Document>.summarizer_token_overlap

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### summarizer_max_text_len
            
Configuration for values:

* summarizer_model = sshleifer/distilbart-cnn-12-6 (default)
* summarizer_token_overlap = 50 (default)
* summarizer_max_text_len = 200 (default)

Can be called using:

    <Document>.x('summarizer_max_text_len')
    # or
    <Document>.summarizer_max_text_len

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### slow_summary
            
Summarize the text using the Huggingface summarization pipeline

Can be called using:

    <Document>.x('slow_summary')
    # or
    <Document>.slow_summary

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### qam_model_id
            
Configuration for values:

* qam_model_id = deepset/minilm-uncased-squad2 (default)

Can be called using:

    <Document>.x('qam_model_id')
    # or
    <Document>.qam_model_id

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### answers
            
Extract answers from the text using the Huggingface question answering pipeline

Can be called using:

    <Document>.x('answers')
    # or
    <Document>.answers

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### chat_model_id
            
In order to use openai-chatgpt, you can use 'gpt-3.5-turbo' or 'gpt-4'.Additionally, we support models used by gpt4all library whichcan be run locally and most are available for commercial purposes. Currently available models are: ['wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0', 'ggml-model-gpt4all-falcon-q4_0', 'ous-hermes-13b.ggmlv3.q4_0', 'GPT4All-13B-snoozy.ggmlv3.q4_0', 'orca-mini-7b.ggmlv3.q4_0', 'orca-mini-3b.ggmlv3.q4_0', 'orca-mini-13b.ggmlv3.q4_0', 'wizardLM-13B-Uncensored.ggmlv3.q4_0', 'ggml-replit-code-v1-3', 'ggml-all-MiniLM-L6-v2-f16', 'starcoderbase-3b-ggml', 'starcoderbase-7b-ggml', 'llama-2-7b-chat.ggmlv3.q4_0']

Can be called using:

    <Document>.x('chat_model_id')
    # or
    <Document>.chat_model_id

return type
: typing.Any

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### chat_answers
            
Extract answers from the text using OpenAI Chat GPT and other models.

Can be called using:

    <Document>.x('chat_answers')
    # or
    <Document>.chat_answers

return type
: typing.Callable[[list[str], list[str] | str], list[str]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, PIL.Image.Image, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, mediawiki, pandoc, text/html, text/markdown, text/rtf

### meta_pdf
            
Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]

Can be called using:

    <Document>.x('meta_pdf')
    # or
    <Document>.meta_pdf

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### pages_bbox
            
Loads the pdf file into a list of [][pydoxtools.document_base.DocumentElement]

Can be called using:

    <Document>.x('pages_bbox')
    # or
    <Document>.pages_bbox

return type
: <class 'numpy.ndarray'\>

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### image_dpi
            
The dpi when rendering the document. The standard image generation resolution is set to 216 dpi for pdfs as we want to have sufficient DPI for downstram OCR tasks (e.g. table extraction)

Can be called using:

    <Document>.x('image_dpi')
    # or
    <Document>.image_dpi

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### images
            
Access images as a dictionary with page numbers as keys for downstream processing tasks

Can be called using:

    <Document>.x('images')
    # or
    <Document>.images

return type
: dict[<module 'PIL.Image' from '/home/tom/.cache/pypoetry/virtualenvs/pydoxtools-ob-vhHEj-py3.10/lib/python3.10/site-packages/PIL/Image.py'\>] | typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_box_levels
            
Extracts the table candidates from the document. As this is an image, we need to use a different method than for pdfs. Right now this relies on neural networks. TODO: add adtitional pure text-based method.

Can be called using:

    <Document>.x('table_box_levels')
    # or
    <Document>.table_box_levels

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_candidates
            
Extracts the table candidates from the document. As this is an image, we need to use a different method than for pdfs. Right now this relies on neural networks. TODO: add adtitional pure text-based method.

Can be called using:

    <Document>.x('table_candidates')
    # or
    <Document>.table_candidates

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### valid_tables
            
Filter valid tables from table candidates by looking if meaningful values can be extracted

Can be called using:

    <Document>.x('valid_tables')
    # or
    <Document>.valid_tables

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_df0
            
Filter valid tables from table candidates by looking if meaningful values can be extracted

Can be called using:

    <Document>.x('table_df0')
    # or
    <Document>.table_df0

return type
: list[pandas.core.frame.DataFrame]

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_areas
            
Areas of all detected tables

Can be called using:

    <Document>.x('table_areas')
    # or
    <Document>.table_areas

return type
: list[numpy.ndarray]

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### titles
            
Extracts the titles from the document by detecting unusual font styles

Can be called using:

    <Document>.x('titles')
    # or
    <Document>.titles

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff, text/html

### side_titles
            
Extracts the titles from the document by detecting unusual font styles

Can be called using:

    <Document>.x('side_titles')
    # or
    <Document>.side_titles

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### table_context
            
Outputs a dictionary with the context of each table in the document

Can be called using:

    <Document>.x('table_context')
    # or
    <Document>.table_context

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, application/pdf, image, image/jpeg, image/png, image/tiff

### html_keywords_str
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('html_keywords_str')
    # or
    <Document>.html_keywords_str

return type
: typing.Any

supports pipeline flows:
: text/html

### main_content_clean_html
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('main_content_clean_html')
    # or
    <Document>.main_content_clean_html

return type
: typing.Any

supports pipeline flows:
: text/html

### summary
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('summary')
    # or
    <Document>.summary

return type
: typing.Any

supports pipeline flows:
: text/html

### goose_article
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('goose_article')
    # or
    <Document>.goose_article

return type
: typing.Any

supports pipeline flows:
: text/html

### main_content
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('main_content')
    # or
    <Document>.main_content

return type
: typing.Any

supports pipeline flows:
: text/html

### schemadata
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('schemadata')
    # or
    <Document>.schemadata

return type
: typing.Any

supports pipeline flows:
: text/html

### final_urls
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('final_urls')
    # or
    <Document>.final_urls

return type
: typing.Any

supports pipeline flows:
: text/html

### pdf_links
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('pdf_links')
    # or
    <Document>.pdf_links

return type
: typing.Any

supports pipeline flows:
: text/html

### title
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('title')
    # or
    <Document>.title

return type
: typing.Any

supports pipeline flows:
: text/html

### short_title
            
Extracts the main content from the html document, removing boilerplate and other noise

Can be called using:

    <Document>.x('short_title')
    # or
    <Document>.short_title

return type
: typing.Any

supports pipeline flows:
: text/html

### urls
            
Extracts the urls from the html document

Can be called using:

    <Document>.x('urls')
    # or
    <Document>.urls

return type
: typing.Any

supports pipeline flows:
: text/html

### main_image
            
Extracts the main image from the html document

Can be called using:

    <Document>.x('main_image')
    # or
    <Document>.main_image

return type
: typing.Any

supports pipeline flows:
: text/html

### html_keywords
            
Extracts explicitly given keywords from the html document

Can be called using:

    <Document>.x('html_keywords')
    # or
    <Document>.html_keywords

return type
: typing.Any

supports pipeline flows:
: text/html

### pandoc_document
            
Loads the document using the pandoc project [https://pandoc.org/](https://pandoc.org/) into a pydoxtools list of [][pydoxtools.document_base.DocumentElement]

Can be called using:

    <Document>.x('pandoc_document')
    # or
    <Document>.pandoc_document

return type
: Pandoc(Meta, [Block])

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### full_text_format
            
The format used to convert the document to a string

Can be called using:

    <Document>.x('full_text_format')
    # or
    <Document>.full_text_format

return type
: typing.Any

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### convert_to
            
Generic pandoc converter for other document formats. TODO: better docs

Can be called using:

    <Document>.x('convert_to')
    # or
    <Document>.convert_to

return type
: typing.Any

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### clean_format
            
The format used to convert the document to a clean string for downstream processing tasks

Can be called using:

    <Document>.x('clean_format')
    # or
    <Document>.clean_format

return type
: typing.Any

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### meta_pandoc
            
meta information from pandoc document

Can be called using:

    <Document>.x('meta_pandoc')
    # or
    <Document>.meta_pandoc

return type
: typing.Any

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### sections
            
Extracts the sections from the document by grouping text elements

Can be called using:

    <Document>.x('sections')
    # or
    <Document>.sections

return type
: typing.Any

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, mediawiki, pandoc, text/markdown, text/rtf

### ocr_lang
            
Configuration for the ocr extractor. We can turn it on/off and specify the language used for OCR.

Can be called using:

    <Document>.x('ocr_lang')
    # or
    <Document>.ocr_lang

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### ocr_on
            
Configuration for the ocr extractor. We can turn it on/off and specify the language used for OCR.

Can be called using:

    <Document>.x('ocr_on')
    # or
    <Document>.ocr_on

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### pil_image
            
Converts the image to a PIL-style image for downstream processing tasks

Can be called using:

    <Document>.x('pil_image')
    # or
    <Document>.pil_image

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### ocr_pdf_file
            
Extracts the text from the document using OCR. It does this by creating a pdf which is important in order to keep the positional information of the text elements.

Can be called using:

    <Document>.x('ocr_pdf_file')
    # or
    <Document>.ocr_pdf_file

return type
: typing.Any

supports pipeline flows:
: PIL.Image.Image, image, image/jpeg, image/png, image/tiff

### data_sel
            
select values by key from source data in Document

Can be called using:

    <Document>.x('data_sel')
    # or
    <Document>.data_sel

return type
: typing.Callable[..., dict]

supports pipeline flows:
: <class 'dict'\>, application/x-yaml

### keys
            
Get the keys of the dictionary

Can be called using:

    <Document>.x('keys')
    # or
    <Document>.keys

return type
: typing.Any

supports pipeline flows:
: <class 'dict'\>, application/x-yaml

### values
            
Get the values of the dictionary

Can be called using:

    <Document>.x('values')
    # or
    <Document>.values

return type
: typing.Any

supports pipeline flows:
: <class 'dict'\>, application/x-yaml

### items
            
Get the items of the dictionary

Can be called using:

    <Document>.x('items')
    # or
    <Document>.items

return type
: typing.Any

supports pipeline flows:
: <class 'dict'\>, application/x-yaml