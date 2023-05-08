# Pipelines

This documents the output values of the nodes of each pipeline that 
can be accessed through the pipeline interface.

Pipeline visualizations for every supported file type can be found
[here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

## [pydoxtools.Document][]

### raw_content
            


Can be called using:

    doc.x('raw_content')
    # or
    doc.raw_content

return type
: bytes | str

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### full_text
            


Can be called using:

    doc.x('full_text')
    # or
    doc.full_text

return type
: <class 'str'>

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### clean_text
            


Can be called using:

    doc.x('clean_text')
    # or
    doc.clean_text

return type
: <class 'str'>

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### data
            


Can be called using:

    doc.x('data')
    # or
    doc.data

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### text_box_elements
            


Can be called using:

    doc.x('text_box_elements')
    # or
    doc.text_box_elements

return type
: <class 'pandas.core.frame.DataFrame'>

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### text_box_list
            


Can be called using:

    doc.x('text_box_list')
    # or
    doc.text_box_list

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### tables_df
            


Can be called using:

    doc.x('tables_df')
    # or
    doc.tables_df

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### tables_dict
            


Can be called using:

    doc.x('tables_dict')
    # or
    doc.tables_dict

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### tables
            


Can be called using:

    doc.x('tables')
    # or
    doc.tables

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### addresses
            


Can be called using:

    doc.x('addresses')
    # or
    doc.addresses

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### num_pages
            


Can be called using:

    doc.x('num_pages')
    # or
    doc.num_pages

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### num_words
            


Can be called using:

    doc.x('num_words')
    # or
    doc.num_words

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### num_sents
            


Can be called using:

    doc.x('num_sents')
    # or
    doc.num_sents

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### a_d_ratio
            


Can be called using:

    doc.x('a_d_ratio')
    # or
    doc.a_d_ratio

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### language
            


Can be called using:

    doc.x('language')
    # or
    doc.language

return type
: <class 'str'>

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_model_size
            


Can be called using:

    doc.x('spacy_model_size')
    # or
    doc.spacy_model_size

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_model
            


Can be called using:

    doc.x('spacy_model')
    # or
    doc.spacy_model

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_doc
            


Can be called using:

    doc.x('spacy_doc')
    # or
    doc.spacy_doc

return type
: dict[str, typing.Any]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_nlp
            


Can be called using:

    doc.x('spacy_nlp')
    # or
    doc.spacy_nlp

return type
: dict[str, typing.Any]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_vectors
            


Can be called using:

    doc.x('spacy_vectors')
    # or
    doc.spacy_vectors

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_embeddings
            


Can be called using:

    doc.x('spacy_embeddings')
    # or
    doc.spacy_embeddings

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_sents
            


Can be called using:

    doc.x('spacy_sents')
    # or
    doc.spacy_sents

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### spacy_noun_chunks
            


Can be called using:

    doc.x('spacy_noun_chunks')
    # or
    doc.spacy_noun_chunks

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### entities
            


Can be called using:

    doc.x('entities')
    # or
    doc.entities

return type
: dict[str, list[str]]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### url
            


Can be called using:

    doc.x('url')
    # or
    doc.url

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sents
            


Can be called using:

    doc.x('sents')
    # or
    doc.sents

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_chunks
            


Can be called using:

    doc.x('noun_chunks')
    # or
    doc.noun_chunks

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vector
            


Can be called using:

    doc.x('vector')
    # or
    doc.vector

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sent_vecs
            


Can be called using:

    doc.x('sent_vecs')
    # or
    doc.sent_vecs

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sent_ids
            


Can be called using:

    doc.x('sent_ids')
    # or
    doc.sent_ids

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_vecs
            


Can be called using:

    doc.x('noun_vecs')
    # or
    doc.noun_vecs

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_ids
            


Can be called using:

    doc.x('noun_ids')
    # or
    doc.noun_ids

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vectorizer_model
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    doc.x('vectorizer_model')
    # or
    doc.vectorizer_model

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vectorizer_only_tokenizer
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    doc.x('vectorizer_only_tokenizer')
    # or
    doc.vectorizer_only_tokenizer

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vectorizer_overlap_ratio
            
Choose the embeddings model (huggingface-style) and if we wantto do the vectorization using only the tokenizer. Using only thetokenizer is MUCH faster and uses lower CPU than creating actualcontextual embeddings using the model. BUt is also lower qualitybecause it lacks the context.

Can be called using:

    doc.x('vectorizer_overlap_ratio')
    # or
    doc.vectorizer_overlap_ratio

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vec_res
            


Can be called using:

    doc.x('vec_res')
    # or
    doc.vec_res

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### tok_embeddings
            


Can be called using:

    doc.x('tok_embeddings')
    # or
    doc.tok_embeddings

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### tokens
            


Can be called using:

    doc.x('tokens')
    # or
    doc.tokens

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### embedding
            


Can be called using:

    doc.x('embedding')
    # or
    doc.embedding

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### min_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    doc.x('min_size_text_segment')
    # or
    doc.min_size_text_segment

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### max_size_text_segment
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    doc.x('max_size_text_segment')
    # or
    doc.max_size_text_segment

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### text_segment_overlap
            
controls the text segmentation for knowledge basesoverlap is only relevant for large text segmenets that need tobe split up into smaller pieces.

Can be called using:

    doc.x('text_segment_overlap')
    # or
    doc.text_segment_overlap

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### text_segments
            


Can be called using:

    doc.x('text_segments')
    # or
    doc.text_segments

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### text_segment_vectors
            


Can be called using:

    doc.x('text_segment_vectors')
    # or
    doc.text_segment_vectors

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_index
            


Can be called using:

    doc.x('noun_index')
    # or
    doc.noun_index

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### vectorizer
            


Can be called using:

    doc.x('vectorizer')
    # or
    doc.vectorizer

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_query
            


Can be called using:

    doc.x('noun_query')
    # or
    doc.noun_query

return type
: typing.Callable

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### noun_graph
            


Can be called using:

    doc.x('noun_graph')
    # or
    doc.noun_graph

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### top_k_text_rank_keywords
            


Can be called using:

    doc.x('top_k_text_rank_keywords')
    # or
    doc.top_k_text_rank_keywords

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### textrank_keywords
            


Can be called using:

    doc.x('textrank_keywords')
    # or
    doc.textrank_keywords

return type
: set[str]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### keywords
            


Can be called using:

    doc.x('keywords')
    # or
    doc.keywords

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sent_index
            


Can be called using:

    doc.x('sent_index')
    # or
    doc.sent_index

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sent_query
            


Can be called using:

    doc.x('sent_query')
    # or
    doc.sent_query

return type
: typing.Callable

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### sent_graph
            


Can be called using:

    doc.x('sent_graph')
    # or
    doc.sent_graph

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### top_k_text_rank_sentences
            


Can be called using:

    doc.x('top_k_text_rank_sentences')
    # or
    doc.top_k_text_rank_sentences

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### textrank_sents
            


Can be called using:

    doc.x('textrank_sents')
    # or
    doc.textrank_sents

return type
: set[str]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### summarizer_model
            


Can be called using:

    doc.x('summarizer_model')
    # or
    doc.summarizer_model

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### summarizer_token_overlap
            


Can be called using:

    doc.x('summarizer_token_overlap')
    # or
    doc.summarizer_token_overlap

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### summarizer_max_text_len
            


Can be called using:

    doc.x('summarizer_max_text_len')
    # or
    doc.summarizer_max_text_len

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### slow_summary
            


Can be called using:

    doc.x('slow_summary')
    # or
    doc.slow_summary

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### qam_model_id
            


Can be called using:

    doc.x('qam_model_id')
    # or
    doc.qam_model_id

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### answers
            


Can be called using:

    doc.x('answers')
    # or
    doc.answers

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### openai_chat_model_id
            


Can be called using:

    doc.x('openai_chat_model_id')
    # or
    doc.openai_chat_model_id

return type
: 

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### chat_answers
            


Can be called using:

    doc.x('chat_answers')
    # or
    doc.chat_answers

return type
: typing.Callable[[list[str], list[str] | str], list[str]]

supports pipelines
: *,<class 'dict'>,application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/x-yaml,image,image/jpeg,image/png,image/tiff,pandoc,text/html,text/markdown,text/rtf

### page_set
            


Can be called using:

    doc.x('page_set')
    # or
    doc.page_set

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### pages_bbox
            


Can be called using:

    doc.x('pages_bbox')
    # or
    doc.pages_bbox

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### elements
            


Can be called using:

    doc.x('elements')
    # or
    doc.elements

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### meta
            


Can be called using:

    doc.x('meta')
    # or
    doc.meta

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### line_elements
            


Can be called using:

    doc.x('line_elements')
    # or
    doc.line_elements

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### graphic_elements
            


Can be called using:

    doc.x('graphic_elements')
    # or
    doc.graphic_elements

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### lists
            


Can be called using:

    doc.x('lists')
    # or
    doc.lists

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: application/epub+zip,application/pdf,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image,image/jpeg,image/png,image/tiff,pandoc,text/markdown,text/rtf

### table_box_levels
            


Can be called using:

    doc.x('table_box_levels')
    # or
    doc.table_box_levels

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### table_candidates
            


Can be called using:

    doc.x('table_candidates')
    # or
    doc.table_candidates

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### table_df0
            


Can be called using:

    doc.x('table_df0')
    # or
    doc.table_df0

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### titles
            


Can be called using:

    doc.x('titles')
    # or
    doc.titles

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff,text/html

### side_titles
            


Can be called using:

    doc.x('side_titles')
    # or
    doc.side_titles

return type
: 

supports pipelines
: application/pdf,image,image/jpeg,image/png,image/tiff

### html_keywords_str
            


Can be called using:

    doc.x('html_keywords_str')
    # or
    doc.html_keywords_str

return type
: 

supports pipelines
: text/html

### main_content_clean_html
            


Can be called using:

    doc.x('main_content_clean_html')
    # or
    doc.main_content_clean_html

return type
: 

supports pipelines
: text/html

### summary
            


Can be called using:

    doc.x('summary')
    # or
    doc.summary

return type
: 

supports pipelines
: text/html

### goose_article
            


Can be called using:

    doc.x('goose_article')
    # or
    doc.goose_article

return type
: 

supports pipelines
: text/html

### main_content
            


Can be called using:

    doc.x('main_content')
    # or
    doc.main_content

return type
: 

supports pipelines
: text/html

### schemadata
            


Can be called using:

    doc.x('schemadata')
    # or
    doc.schemadata

return type
: 

supports pipelines
: text/html

### final_urls
            


Can be called using:

    doc.x('final_urls')
    # or
    doc.final_urls

return type
: 

supports pipelines
: text/html

### pdf_links
            


Can be called using:

    doc.x('pdf_links')
    # or
    doc.pdf_links

return type
: 

supports pipelines
: text/html

### title
            


Can be called using:

    doc.x('title')
    # or
    doc.title

return type
: 

supports pipelines
: text/html

### short_title
            


Can be called using:

    doc.x('short_title')
    # or
    doc.short_title

return type
: 

supports pipelines
: text/html

### urls
            


Can be called using:

    doc.x('urls')
    # or
    doc.urls

return type
: 

supports pipelines
: text/html

### main_image
            


Can be called using:

    doc.x('main_image')
    # or
    doc.main_image

return type
: 

supports pipelines
: text/html

### html_keywords
            


Can be called using:

    doc.x('html_keywords')
    # or
    doc.html_keywords

return type
: 

supports pipelines
: text/html

### pandoc_document
            


Can be called using:

    doc.x('pandoc_document')
    # or
    doc.pandoc_document

return type
: Pandoc(Meta, [Block])

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### full_text_format
            


Can be called using:

    doc.x('full_text_format')
    # or
    doc.full_text_format

return type
: 

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### convert_to
            


Can be called using:

    doc.x('convert_to')
    # or
    doc.convert_to

return type
: 

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### clean_format
            


Can be called using:

    doc.x('clean_format')
    # or
    doc.clean_format

return type
: 

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### sections
            


Can be called using:

    doc.x('sections')
    # or
    doc.sections

return type
: 

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### pandoc_blocks
            


Can be called using:

    doc.x('pandoc_blocks')
    # or
    doc.pandoc_blocks

return type
: list['pandoc.types.Block']

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### headers
            


Can be called using:

    doc.x('headers')
    # or
    doc.headers

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: application/epub+zip,application/vnd.oasis.opendocument.text,application/vnd.openxmlformats-officedocument.wordprocessingml.document,pandoc,text/markdown,text/rtf

### ocr_lang
            


Can be called using:

    doc.x('ocr_lang')
    # or
    doc.ocr_lang

return type
: 

supports pipelines
: image,image/jpeg,image/png,image/tiff

### ocr_on
            


Can be called using:

    doc.x('ocr_on')
    # or
    doc.ocr_on

return type
: 

supports pipelines
: image,image/jpeg,image/png,image/tiff

### ocr_pdf_file
            


Can be called using:

    doc.x('ocr_pdf_file')
    # or
    doc.ocr_pdf_file

return type
: 

supports pipelines
: image,image/jpeg,image/png,image/tiff

### data_sel
            
select values by key from source data in Document

Can be called using:

    doc.x('data_sel')
    # or
    doc.data_sel

return type
: typing.Callable[..., dict]

supports pipelines
: <class 'dict'>,application/x-yaml

### keys
            


Can be called using:

    doc.x('keys')
    # or
    doc.keys

return type
: 

supports pipelines
: <class 'dict'>,application/x-yaml

### values
            


Can be called using:

    doc.x('values')
    # or
    doc.values

return type
: 

supports pipelines
: <class 'dict'>,application/x-yaml

### items
            


Can be called using:

    doc.x('items')
    # or
    doc.items

return type
: 

supports pipelines
: <class 'dict'>,application/x-yaml

## [pydoxtools.DocumentBag][]

### doc_configuration
            
We can pass through a configuration object to Documents that are created in our document bag. Any setting that is supported by Document can be specified here.

Can be called using:

    doc.x('doc_configuration')
    # or
    doc.doc_configuration

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### docs
            
create a bag with one document for each file that was foundFrom this point we can hand off the logic to str(Bag) pipeline.

Can be called using:

    doc.x('docs')
    # or
    doc.docs

return type
: <class 'dask.bag.core.Bag'>

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### take
            


Can be called using:

    doc.x('take')
    # or
    doc.take

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### compute
            


Can be called using:

    doc.x('compute')
    # or
    doc.compute

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### get_dicts
            


Can be called using:

    doc.x('get_dicts')
    # or
    doc.get_dicts

return type
: typing.Callable[[typing.Any], dask.bag.core.Bag]

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### e
            


Can be called using:

    doc.x('e')
    # or
    doc.e

return type
: typing.Callable[..., pydoxtools.document.DocumentBag]

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### idx_dict
            


Can be called using:

    doc.x('idx_dict')
    # or
    doc.idx_dict

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### vectorizer
            


Can be called using:

    doc.x('vectorizer')
    # or
    doc.vectorizer

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### chroma_index
            
in order to build an index in chrome db we need a key, text, embeddings and a key. Those come from a daskbag with dictionaries with those keys. Caching is important here in order to retain the index

Can be called using:

    doc.x('chroma_index')
    # or
    doc.chroma_index

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### compute_index
            
in order to build an index in chrome db we need a key, text, embeddings and a key. Those come from a daskbag with dictionaries with those keys. Caching is important here in order to retain the index

Can be called using:

    doc.x('compute_index')
    # or
    doc.compute_index

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### query_chroma
            
in order to build an index in chrome db we need a key, text, embeddings and a key. Those come from a daskbag with dictionaries with those keys. Caching is important here in order to retain the index

Can be called using:

    doc.x('query_chroma')
    # or
    doc.query_chroma

return type
: 

supports pipelines
: <class 'dask.bag.core.Bag'>,<class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### sql
            


Can be called using:

    doc.x('sql')
    # or
    doc.sql

return type
: 

supports pipelines
: <class 'pydoxtools.document.DatabaseSource'>

### connection_string
            


Can be called using:

    doc.x('connection_string')
    # or
    doc.connection_string

return type
: 

supports pipelines
: <class 'pydoxtools.document.DatabaseSource'>

### index_column
            


Can be called using:

    doc.x('index_column')
    # or
    doc.index_column

return type
: 

supports pipelines
: <class 'pydoxtools.document.DatabaseSource'>

### bytes_per_chunk
            


Can be called using:

    doc.x('bytes_per_chunk')
    # or
    doc.bytes_per_chunk

return type
: 

supports pipelines
: <class 'pydoxtools.document.DatabaseSource'>

### dataframe
            


Can be called using:

    doc.x('dataframe')
    # or
    doc.dataframe

return type
: <module 'dask.dataframe' from '/home/tom/.cache/pypoetry/virtualenvs/componardo-yE4zvXcb-py3.10/lib/python3.10/site-packages/dask/dataframe/__init__.py'>

supports pipelines
: <class 'pydoxtools.document.DatabaseSource'>

### bag
            
create a dask bag with all the filepaths in it

Can be called using:

    doc.x('bag')
    # or
    doc.bag

return type
: 

supports pipelines
: <class 'list'>,<class 'pathlib.Path'>,<class 'pydoxtools.document.DatabaseSource'>

### root_path
            


Can be called using:

    doc.x('root_path')
    # or
    doc.root_path

return type
: 

supports pipelines
: <class 'list'>,<class 'pathlib.Path'>

### file_path_list
            


Can be called using:

    doc.x('file_path_list')
    # or
    doc.file_path_list

return type
: <class 'dask.bag.core.Bag'>list[pathlib.Path]

supports pipelines
: <class 'list'>,<class 'pathlib.Path'>

### dir_list
            


Can be called using:

    doc.x('dir_list')
    # or
    doc.dir_list

return type
: <class 'dask.bag.core.Bag'>list[pathlib.Path]

supports pipelines
: <class 'list'>,<class 'pathlib.Path'>