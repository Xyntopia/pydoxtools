# Pipelines

This documents the output values of the nodes of each pipeline that 
can be accessed through the pipeline interface.

## [pydoxtools.Document][]

### raw_content

Can be called using:

    doc.x('raw_content')
    # or
    doc.raw_content

return type
: bytes | str

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### full_text

Can be called using:

    doc.x('full_text')
    # or
    doc.full_text

return type
: <class 'str'>

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### clean_text

Can be called using:

    doc.x('clean_text')
    # or
    doc.clean_text

return type
: <class 'str'>

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_box_elements

Can be called using:

    doc.x('text_box_elements')
    # or
    doc.text_box_elements

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_box_list

Can be called using:

    doc.x('text_box_list')
    # or
    doc.text_box_list

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### tables_df

Can be called using:

    doc.x('tables_df')
    # or
    doc.tables_df

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### tables_dict

Can be called using:

    doc.x('tables_dict')
    # or
    doc.tables_dict

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### tables

Can be called using:

    doc.x('tables')
    # or
    doc.tables

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### addresses

Can be called using:

    doc.x('addresses')
    # or
    doc.addresses

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### num_pages

Can be called using:

    doc.x('num_pages')
    # or
    doc.num_pages

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### num_words

Can be called using:

    doc.x('num_words')
    # or
    doc.num_words

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### num_sents

Can be called using:

    doc.x('num_sents')
    # or
    doc.num_sents

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### a_d_ratio

Can be called using:

    doc.x('a_d_ratio')
    # or
    doc.a_d_ratio

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### language

Can be called using:

    doc.x('language')
    # or
    doc.language

return type
: <class 'str'>

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_model_size

Can be called using:

    doc.x('spacy_model_size')
    # or
    doc.spacy_model_size

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_model

Can be called using:

    doc.x('spacy_model')
    # or
    doc.spacy_model

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_doc

Can be called using:

    doc.x('spacy_doc')
    # or
    doc.spacy_doc

return type
: dict[str, typing.Any]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_nlp

Can be called using:

    doc.x('spacy_nlp')
    # or
    doc.spacy_nlp

return type
: dict[str, typing.Any]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_vectors

Can be called using:

    doc.x('spacy_vectors')
    # or
    doc.spacy_vectors

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_embeddings

Can be called using:

    doc.x('spacy_embeddings')
    # or
    doc.spacy_embeddings

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_sents

Can be called using:

    doc.x('spacy_sents')
    # or
    doc.spacy_sents

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### spacy_noun_chunks

Can be called using:

    doc.x('spacy_noun_chunks')
    # or
    doc.spacy_noun_chunks

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### entities

Can be called using:

    doc.x('entities')
    # or
    doc.entities

return type
: dict[str, list[str]]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### url

Can be called using:

    doc.x('url')
    # or
    doc.url

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sents

Can be called using:

    doc.x('sents')
    # or
    doc.sents

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_chunks

Can be called using:

    doc.x('noun_chunks')
    # or
    doc.noun_chunks

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### vector

Can be called using:

    doc.x('vector')
    # or
    doc.vector

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sent_vecs

Can be called using:

    doc.x('sent_vecs')
    # or
    doc.sent_vecs

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sent_ids

Can be called using:

    doc.x('sent_ids')
    # or
    doc.sent_ids

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_vecs

Can be called using:

    doc.x('noun_vecs')
    # or
    doc.noun_vecs

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_ids

Can be called using:

    doc.x('noun_ids')
    # or
    doc.noun_ids

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_segments

Can be called using:

    doc.x('text_segments')
    # or
    doc.text_segments

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_segment_model

Can be called using:

    doc.x('text_segment_model')
    # or
    doc.text_segment_model

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_segment_only_tokenizer

Can be called using:

    doc.x('text_segment_only_tokenizer')
    # or
    doc.text_segment_only_tokenizer

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### text_segment_vectors

Can be called using:

    doc.x('text_segment_vectors')
    # or
    doc.text_segment_vectors

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_index

Can be called using:

    doc.x('noun_index')
    # or
    doc.noun_index

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### vectorizer

Can be called using:

    doc.x('vectorizer')
    # or
    doc.vectorizer

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_query

Can be called using:

    doc.x('noun_query')
    # or
    doc.noun_query

return type
: typing.Callable

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### noun_graph

Can be called using:

    doc.x('noun_graph')
    # or
    doc.noun_graph

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### top_k_text_rank_keywords

Can be called using:

    doc.x('top_k_text_rank_keywords')
    # or
    doc.top_k_text_rank_keywords

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### textrank_keywords

Can be called using:

    doc.x('textrank_keywords')
    # or
    doc.textrank_keywords

return type
: set[str]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### keywords

Can be called using:

    doc.x('keywords')
    # or
    doc.keywords

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sent_index

Can be called using:

    doc.x('sent_index')
    # or
    doc.sent_index

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sent_query

Can be called using:

    doc.x('sent_query')
    # or
    doc.sent_query

return type
: typing.Callable

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### sent_graph

Can be called using:

    doc.x('sent_graph')
    # or
    doc.sent_graph

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### top_k_text_rank_sentences

Can be called using:

    doc.x('top_k_text_rank_sentences')
    # or
    doc.top_k_text_rank_sentences

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### textrank_sents

Can be called using:

    doc.x('textrank_sents')
    # or
    doc.textrank_sents

return type
: set[str]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### summarizer_model

Can be called using:

    doc.x('summarizer_model')
    # or
    doc.summarizer_model

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### summarizer_token_overlap

Can be called using:

    doc.x('summarizer_token_overlap')
    # or
    doc.summarizer_token_overlap

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### summarizer_max_text_len

Can be called using:

    doc.x('summarizer_max_text_len')
    # or
    doc.summarizer_max_text_len

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### summary

Can be called using:

    doc.x('summary')
    # or
    doc.summary

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### qam_model_id

Can be called using:

    doc.x('qam_model_id')
    # or
    doc.qam_model_id

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### answers

Can be called using:

    doc.x('answers')
    # or
    doc.answers

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### openai_chat_model_id

Can be called using:

    doc.x('openai_chat_model_id')
    # or
    doc.openai_chat_model_id

return type
: 

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### chat_answers

Can be called using:

    doc.x('chat_answers')
    # or
    doc.chat_answers

return type
: typing.Callable[[list[str], list[str] | str], list[str]]

supports pipelines
: .pdf,.odt,.html,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,.yaml,dict,image,*,.epub,.jpg

### page_set

Can be called using:

    doc.x('page_set')
    # or
    doc.page_set

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### pages_bbox

Can be called using:

    doc.x('pages_bbox')
    # or
    doc.pages_bbox

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### elements

Can be called using:

    doc.x('elements')
    # or
    doc.elements

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### meta

Can be called using:

    doc.x('meta')
    # or
    doc.meta

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### line_elements

Can be called using:

    doc.x('line_elements')
    # or
    doc.line_elements

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### graphic_elements

Can be called using:

    doc.x('graphic_elements')
    # or
    doc.graphic_elements

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### lists

Can be called using:

    doc.x('lists')
    # or
    doc.lists

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: .pdf,.odt,.jpeg,.markdown,.md,.docx,.tif,.rtf,pandoc,.png,.tiff,image,.epub,.jpg

### table_box_levels

Can be called using:

    doc.x('table_box_levels')
    # or
    doc.table_box_levels

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### table_candidates

Can be called using:

    doc.x('table_candidates')
    # or
    doc.table_candidates

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### table_df0

Can be called using:

    doc.x('table_df0')
    # or
    doc.table_df0

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### titles

Can be called using:

    doc.x('titles')
    # or
    doc.titles

return type
: 

supports pipelines
: .pdf,.html,.jpeg,.tif,.png,.tiff,image,.jpg

### side_titles

Can be called using:

    doc.x('side_titles')
    # or
    doc.side_titles

return type
: 

supports pipelines
: .pdf,.jpeg,.tif,.png,.tiff,image,.jpg

### html_keywords_str

Can be called using:

    doc.x('html_keywords_str')
    # or
    doc.html_keywords_str

return type
: 

supports pipelines
: .html

### main_content_clean_html

Can be called using:

    doc.x('main_content_clean_html')
    # or
    doc.main_content_clean_html

return type
: 

supports pipelines
: .html

### goose_article

Can be called using:

    doc.x('goose_article')
    # or
    doc.goose_article

return type
: 

supports pipelines
: .html

### main_content

Can be called using:

    doc.x('main_content')
    # or
    doc.main_content

return type
: 

supports pipelines
: .html

### schemadata

Can be called using:

    doc.x('schemadata')
    # or
    doc.schemadata

return type
: 

supports pipelines
: .html

### final_urls

Can be called using:

    doc.x('final_urls')
    # or
    doc.final_urls

return type
: 

supports pipelines
: .html

### pdf_links

Can be called using:

    doc.x('pdf_links')
    # or
    doc.pdf_links

return type
: 

supports pipelines
: .html

### title

Can be called using:

    doc.x('title')
    # or
    doc.title

return type
: 

supports pipelines
: .html

### short_title

Can be called using:

    doc.x('short_title')
    # or
    doc.short_title

return type
: 

supports pipelines
: .html

### urls

Can be called using:

    doc.x('urls')
    # or
    doc.urls

return type
: 

supports pipelines
: .html

### main_image

Can be called using:

    doc.x('main_image')
    # or
    doc.main_image

return type
: 

supports pipelines
: .html

### html_keywords

Can be called using:

    doc.x('html_keywords')
    # or
    doc.html_keywords

return type
: 

supports pipelines
: .html

### pandoc_document

Can be called using:

    doc.x('pandoc_document')
    # or
    doc.pandoc_document

return type
: Pandoc(Meta, [Block])

supports pipelines
: .odt,.markdown,.md,.docx,.rtf,pandoc,.epub

### output_format

Can be called using:

    doc.x('output_format')
    # or
    doc.output_format

return type
: 

supports pipelines
: .odt,.markdown,.md,.docx,.rtf,pandoc,.epub

### clean_format

Can be called using:

    doc.x('clean_format')
    # or
    doc.clean_format

return type
: 

supports pipelines
: .odt,.markdown,.md,.docx,.rtf,pandoc,.epub

### pandoc_blocks

Can be called using:

    doc.x('pandoc_blocks')
    # or
    doc.pandoc_blocks

return type
: list['pandoc.types.Block']

supports pipelines
: .odt,.markdown,.md,.docx,.rtf,pandoc,.epub

### headers

Can be called using:

    doc.x('headers')
    # or
    doc.headers

return type
: str | list[str] | list[pandas.core.frame.DataFrame]

supports pipelines
: .odt,.markdown,.md,.docx,.rtf,pandoc,.epub

### ocr_lang

Can be called using:

    doc.x('ocr_lang')
    # or
    doc.ocr_lang

return type
: 

supports pipelines
: .jpeg,.tif,.png,.tiff,image,.jpg

### ocr_on

Can be called using:

    doc.x('ocr_on')
    # or
    doc.ocr_on

return type
: 

supports pipelines
: .jpeg,.tif,.png,.tiff,image,.jpg

### ocr_pdf_file

Can be called using:

    doc.x('ocr_pdf_file')
    # or
    doc.ocr_pdf_file

return type
: 

supports pipelines
: .jpeg,.tif,.png,.tiff,image,.jpg

### data

Can be called using:

    doc.x('data')
    # or
    doc.data

return type
: 

supports pipelines
: .yaml,dict