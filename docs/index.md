# Welcome to Pydoxtools Documentation!

For a short overview over Pydoxtools, checkout the readme on the project page:

[Readme](readme_cp)

## Introduction

Pydoxtools provides a user-friendly interface for document analysis and 
manipulation, consisting of three main classes:

- [pydoxtools.Document][]
- [pydoxtools.DocumentBag][]
- [pydoxtools.Pipeline][]

Additionally, it offers a collection of operators:

- [pydoxtools.operators][]

### Analyzing Documents

Both, Document and DocumentBag utilize [pydoxtools.Pipeline][] to define a 
sophisticated pipeline for extracting data from individual or multiple documents. 
You can find a list of all the built-in features for each pipeline here:

-> [pydoxtools.Document][] and [pydoxtools.DocumentBag][] 

To ensure seamless operation, Pydoxtools is designed so that 
[Document][pydoxtools.Document] and [DocumentBag][pydoxtools.DocumentBag] 
automatically organize information in a logical manner while minimizing 
memory and CPU usage. This approach makes the library highly compatible 
with AI and LLMs in automated settings. As a result, it is not possible 
to configure *how* documents are loaded using configuration parameters. 
However, you can easily achieve specific data organization by chaining documents together.

TODO:  provide an example

### Building Custom Pipelines with LLMs (Large Language Models) and other AI Tools

The Pipeline class allows you to create complex, custom pipelines that come
with several built-in features, making them easy to integrate with modern AI tools:

- Mix, extend, or (partially) overwrite pipelines
- Export/import data (yaml, json, python-dict)
- Configure and optimize pipelines
- Convert data into [pydoxtools.Document][] and [pydoxtools.DocumentBag][]

To develop a custom pipeline, you can utilize the extensive library of
[pydoxtools.operators][]. It is generally recommended to use 
[pydoxtools.Document][] or [pydoxtools.DocumentBag][] as a base for 
a new pipeline and only replace small parts to achieve the desired 
custom functionality.

## Visualizing Pipelines

Visualizing pipelines can be incredibly helpful when developing your 
own pipeline on top of a complex one, such as the document pipeline. 
You can visualize the extraction logic for different file types from the Document
class (which is a [pydoxtools.Pipeline][]  itself) as follows:

    doc = Document(fobj=make_path_absolute("./data/demo.docx"))
    # for the currently loaded file type:
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_docx.svg")
    # for the 
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_png.svg", document_logic_id=".png")

This allows you to generate pipelines for various file types:

- [docx](images/document_logic_.docx.svg)
- [png](images/document_logic_.png.svg)
  (click on links to open the images!)

You can find pipelines for every supported file type 
[here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

This feature is also available for custom pipelines!

To learn more, continue to: [Reference](reference)