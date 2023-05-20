# [pydoxtools.Document][]

::: pydoxtools.Document

## Text extraction attributes and functions

The [pydoxtools.Document][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

### raw_content
            


Can be called using:

    <Document>.x('raw_content')
    # or
    <Document>.raw_content

return type
: bytes | str

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### data
            


Can be called using:

    <Document>.x('data')
    # or
    <Document>.data

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### full_text
            


Can be called using:

    <Document>.x('full_text')
    # or
    <Document>.full_text

return type
: <class 'str'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### clean_text
            


Can be called using:

    <Document>.x('clean_text')
    # or
    <Document>.clean_text

return type
: <class 'str'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### meta
            


Can be called using:

    <Document>.x('meta')
    # or
    <Document>.meta

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### file_meta
            
some fast-to-calculate metadata information about a file

Can be called using:

    <Document>.x('file_meta')
    # or
    <Document>.file_meta

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### text_box_elements
            


Can be called using:

    <Document>.x('text_box_elements')
    # or
    <Document>.text_box_elements

return type
: <class 'pandas.core.frame.DataFrame'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### text_box_list
            


Can be called using:

    <Document>.x('text_box_list')
    # or
    <Document>.text_box_list

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### tables_df
            


Can be called using:

    <Document>.x('tables_df')
    # or
    <Document>.tables_df

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### tables_dict
            


Can be called using:

    <Document>.x('tables_dict')
    # or
    <Document>.tables_dict

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### tables
            


Can be called using:

    <Document>.x('tables')
    # or
    <Document>.tables

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### addresses
            


Can be called using:

    <Document>.x('addresses')
    # or
    <Document>.addresses

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### num_pages
            


Can be called using:

    <Document>.x('num_pages')
    # or
    <Document>.num_pages

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### num_words
            


Can be called using:

    <Document>.x('num_words')
    # or
    <Document>.num_words

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### num_sents
            


Can be called using:

    <Document>.x('num_sents')
    # or
    <Document>.num_sents

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### a_d_ratio
            


Can be called using:

    <Document>.x('a_d_ratio')
    # or
    <Document>.a_d_ratio

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### language
            


Can be called using:

    <Document>.x('language')
    # or
    <Document>.language

return type
: <class 'str'\>

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_model_size
            


Can be called using:

    <Document>.x('spacy_model_size')
    # or
    <Document>.spacy_model_size

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_model
            


Can be called using:

    <Document>.x('spacy_model')
    # or
    <Document>.spacy_model

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_doc
            


Can be called using:

    <Document>.x('spacy_doc')
    # or
    <Document>.spacy_doc

return type
: dict[str, typing.Any]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_nlp
            


Can be called using:

    <Document>.x('spacy_nlp')
    # or
    <Document>.spacy_nlp

return type
: dict[str, typing.Any]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_vectors
            


Can be called using:

    <Document>.x('spacy_vectors')
    # or
    <Document>.spacy_vectors

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_embeddings
            


Can be called using:

    <Document>.x('spacy_embeddings')
    # or
    <Document>.spacy_embeddings

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_sents
            


Can be called using:

    <Document>.x('spacy_sents')
    # or
    <Document>.spacy_sents

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### spacy_noun_chunks
            
exracts nounchunks from spacy. Will not be cached because it is allin the spacy doc already

Can be called using:

    <Document>.x('spacy_noun_chunks')
    # or
    <Document>.spacy_noun_chunks

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### entities
            


Can be called using:

    <Document>.x('entities')
    # or
    <Document>.entities

return type
: dict[str, list[str]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### url
            


Can be called using:

    <Document>.x('url')
    # or
    <Document>.url

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sents
            


Can be called using:

    <Document>.x('sents')
    # or
    <Document>.sents

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_chunks
            


Can be called using:

    <Document>.x('noun_chunks')
    # or
    <Document>.noun_chunks

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vector
            


Can be called using:

    <Document>.x('vector')
    # or
    <Document>.vector

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sent_vecs
            


Can be called using:

    <Document>.x('sent_vecs')
    # or
    <Document>.sent_vecs

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sent_ids
            


Can be called using:

    <Document>.x('sent_ids')
    # or
    <Document>.sent_ids

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_vecs
            


Can be called using:

    <Document>.x('noun_vecs')
    # or
    <Document>.noun_vecs

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_ids
            


Can be called using:

    <Document>.x('noun_ids')
    # or
    <Document>.noun_ids

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vectorizer_model
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_model')
    # or
    <Document>.vectorizer_model

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vectorizer_only_tokenizer
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_only_tokenizer')
    # or
    <Document>.vectorizer_only_tokenizer

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vectorizer_overlap_ratio
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    <Document>.x('vectorizer_overlap_ratio')
    # or
    <Document>.vectorizer_overlap_ratio

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vec_res
            


Can be called using:

    <Document>.x('vec_res')
    # or
    <Document>.vec_res

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### tok_embeddings
            


Can be called using:

    <Document>.x('tok_embeddings')
    # or
    <Document>.tok_embeddings

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### tokens
            


Can be called using:

    <Document>.x('tokens')
    # or
    <Document>.tokens

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### embedding
            


Can be called using:

    <Document>.x('embedding')
    # or
    <Document>.embedding

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### min_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('min_size_text_segment')
    # or
    <Document>.min_size_text_segment

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### max_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('max_size_text_segment')
    # or
    <Document>.max_size_text_segment

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### text_segment_overlap
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    <Document>.x('text_segment_overlap')
    # or
    <Document>.text_segment_overlap

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### text_segments
            


Can be called using:

    <Document>.x('text_segments')
    # or
    <Document>.text_segments

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### text_segment_vectors
            


Can be called using:

    <Document>.x('text_segment_vectors')
    # or
    <Document>.text_segment_vectors

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_index
            


Can be called using:

    <Document>.x('noun_index')
    # or
    <Document>.noun_index

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### vectorizer
            


Can be called using:

    <Document>.x('vectorizer')
    # or
    <Document>.vectorizer

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_query
            


Can be called using:

    <Document>.x('noun_query')
    # or
    <Document>.noun_query

return type
: typing.Callable

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### noun_graph
            


Can be called using:

    <Document>.x('noun_graph')
    # or
    <Document>.noun_graph

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### top_k_text_rank_keywords
            


Can be called using:

    <Document>.x('top_k_text_rank_keywords')
    # or
    <Document>.top_k_text_rank_keywords

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### textrank_keywords
            


Can be called using:

    <Document>.x('textrank_keywords')
    # or
    <Document>.textrank_keywords

return type
: set[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### keywords
            


Can be called using:

    <Document>.x('keywords')
    # or
    <Document>.keywords

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sent_index
            


Can be called using:

    <Document>.x('sent_index')
    # or
    <Document>.sent_index

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sent_query
            


Can be called using:

    <Document>.x('sent_query')
    # or
    <Document>.sent_query

return type
: typing.Callable

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### sent_graph
            


Can be called using:

    <Document>.x('sent_graph')
    # or
    <Document>.sent_graph

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### top_k_text_rank_sentences
            


Can be called using:

    <Document>.x('top_k_text_rank_sentences')
    # or
    <Document>.top_k_text_rank_sentences

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### textrank_sents
            


Can be called using:

    <Document>.x('textrank_sents')
    # or
    <Document>.textrank_sents

return type
: set[str]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### summarizer_model
            


Can be called using:

    <Document>.x('summarizer_model')
    # or
    <Document>.summarizer_model

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### summarizer_token_overlap
            


Can be called using:

    <Document>.x('summarizer_token_overlap')
    # or
    <Document>.summarizer_token_overlap

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### summarizer_max_text_len
            


Can be called using:

    <Document>.x('summarizer_max_text_len')
    # or
    <Document>.summarizer_max_text_len

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### slow_summary
            


Can be called using:

    <Document>.x('slow_summary')
    # or
    <Document>.slow_summary

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### qam_model_id
            


Can be called using:

    <Document>.x('qam_model_id')
    # or
    <Document>.qam_model_id

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### answers
            


Can be called using:

    <Document>.x('answers')
    # or
    <Document>.answers

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### openai_chat_model_id
            


Can be called using:

    <Document>.x('openai_chat_model_id')
    # or
    <Document>.openai_chat_model_id

return type
: 

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### chat_answers
            


Can be called using:

    <Document>.x('chat_answers')
    # or
    <Document>.chat_answers

return type
: typing.Callable[[list[str], list[str] | str], list[str]]

supports pipeline flows:
: *, <class 'dict'\>, <class 'list'\>, application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/x-yaml, image, image/jpeg, image/png, image/tiff, pandoc, text/html, text/markdown, text/rtf

### meta_pdf
            


Can be called using:

    <Document>.x('meta_pdf')
    # or
    <Document>.meta_pdf

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### page_set
            


Can be called using:

    <Document>.x('page_set')
    # or
    <Document>.page_set

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### pages_bbox
            


Can be called using:

    <Document>.x('pages_bbox')
    # or
    <Document>.pages_bbox

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### elements
            


Can be called using:

    <Document>.x('elements')
    # or
    <Document>.elements

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### line_elements
            


Can be called using:

    <Document>.x('line_elements')
    # or
    <Document>.line_elements

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### graphic_elements
            


Can be called using:

    <Document>.x('graphic_elements')
    # or
    <Document>.graphic_elements

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### lists
            


Can be called using:

    <Document>.x('lists')
    # or
    <Document>.lists

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipeline flows:
: application/epub+zip, application/pdf, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, image, image/jpeg, image/png, image/tiff, pandoc, text/markdown, text/rtf

### table_box_levels
            


Can be called using:

    <Document>.x('table_box_levels')
    # or
    <Document>.table_box_levels

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### table_candidates
            


Can be called using:

    <Document>.x('table_candidates')
    # or
    <Document>.table_candidates

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### table_df0
            


Can be called using:

    <Document>.x('table_df0')
    # or
    <Document>.table_df0

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### titles
            


Can be called using:

    <Document>.x('titles')
    # or
    <Document>.titles

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff, text/html

### side_titles
            


Can be called using:

    <Document>.x('side_titles')
    # or
    <Document>.side_titles

return type
: 

supports pipeline flows:
: application/pdf, image, image/jpeg, image/png, image/tiff

### html_keywords_str
            


Can be called using:

    <Document>.x('html_keywords_str')
    # or
    <Document>.html_keywords_str

return type
: 

supports pipeline flows:
: text/html

### main_content_clean_html
            


Can be called using:

    <Document>.x('main_content_clean_html')
    # or
    <Document>.main_content_clean_html

return type
: 

supports pipeline flows:
: text/html

### summary
            


Can be called using:

    <Document>.x('summary')
    # or
    <Document>.summary

return type
: 

supports pipeline flows:
: text/html

### goose_article
            


Can be called using:

    <Document>.x('goose_article')
    # or
    <Document>.goose_article

return type
: 

supports pipeline flows:
: text/html

### main_content
            


Can be called using:

    <Document>.x('main_content')
    # or
    <Document>.main_content

return type
: 

supports pipeline flows:
: text/html

### schemadata
            


Can be called using:

    <Document>.x('schemadata')
    # or
    <Document>.schemadata

return type
: 

supports pipeline flows:
: text/html

### final_urls
            


Can be called using:

    <Document>.x('final_urls')
    # or
    <Document>.final_urls

return type
: 

supports pipeline flows:
: text/html

### pdf_links
            


Can be called using:

    <Document>.x('pdf_links')
    # or
    <Document>.pdf_links

return type
: 

supports pipeline flows:
: text/html

### title
            


Can be called using:

    <Document>.x('title')
    # or
    <Document>.title

return type
: 

supports pipeline flows:
: text/html

### short_title
            


Can be called using:

    <Document>.x('short_title')
    # or
    <Document>.short_title

return type
: 

supports pipeline flows:
: text/html

### urls
            


Can be called using:

    <Document>.x('urls')
    # or
    <Document>.urls

return type
: 

supports pipeline flows:
: text/html

### main_image
            


Can be called using:

    <Document>.x('main_image')
    # or
    <Document>.main_image

return type
: 

supports pipeline flows:
: text/html

### html_keywords
            


Can be called using:

    <Document>.x('html_keywords')
    # or
    <Document>.html_keywords

return type
: 

supports pipeline flows:
: text/html

### pandoc_document
            


Can be called using:

    <Document>.x('pandoc_document')
    # or
    <Document>.pandoc_document

return type
: Pandoc(Meta, [Block])

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### full_text_format
            


Can be called using:

    <Document>.x('full_text_format')
    # or
    <Document>.full_text_format

return type
: 

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### convert_to
            


Can be called using:

    <Document>.x('convert_to')
    # or
    <Document>.convert_to

return type
: 

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### clean_format
            


Can be called using:

    <Document>.x('clean_format')
    # or
    <Document>.clean_format

return type
: 

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### sections
            


Can be called using:

    <Document>.x('sections')
    # or
    <Document>.sections

return type
: 

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### pandoc_blocks
            


Can be called using:

    <Document>.x('pandoc_blocks')
    # or
    <Document>.pandoc_blocks

return type
: list['pandoc.types.Block']

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### headers
            


Can be called using:

    <Document>.x('headers')
    # or
    <Document>.headers

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipeline flows:
: application/epub+zip, application/vnd.oasis.opendocument.text, application/vnd.openxmlformats-officedocument.wordprocessingml.document, pandoc, text/markdown, text/rtf

### ocr_lang
            


Can be called using:

    <Document>.x('ocr_lang')
    # or
    <Document>.ocr_lang

return type
: 

supports pipeline flows:
: image, image/jpeg, image/png, image/tiff

### ocr_on
            


Can be called using:

    <Document>.x('ocr_on')
    # or
    <Document>.ocr_on

return type
: 

supports pipeline flows:
: image, image/jpeg, image/png, image/tiff

### ocr_pdf_file
            


Can be called using:

    <Document>.x('ocr_pdf_file')
    # or
    <Document>.ocr_pdf_file

return type
: 

supports pipeline flows:
: image, image/jpeg, image/png, image/tiff

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
            


Can be called using:

    <Document>.x('keys')
    # or
    <Document>.keys

return type
: 

supports pipeline flows:
: <class 'dict'\>, application/x-yaml

### values
            


Can be called using:

    <Document>.x('values')
    # or
    <Document>.values

return type
: 

supports pipeline flows:
: <class 'dict'\>, application/x-yaml

### items
            


Can be called using:

    <Document>.x('items')
    # or
    <Document>.items

return type
: 

supports pipeline flows:
: <class 'dict'\>, application/x-yaml