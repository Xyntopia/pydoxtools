# Welcome to Pydoxtools Documentation!

[Readme](readme_cp)

## Introduction

The main interface that pydoxtools uses are three classes:

- [pydoxtools.Document][]
- [pydoxtools.DocumentSet][]
- [pydoxtools.Pipeline][]

Document & DocumentSet are both predefined [Pipelines][pydoxtools.Pipeline] and
define an extensive pipeline to extract data from individual documents or
a set of documents.

the Pipeline class can be used to build complex, custom pipelines which
have several out-of-the-box features which makes them easy to use in
modern pipelines involving the use of a lot of AI tools:

- they can export/import their data (yaml, json, python-dict),
- they can be configured & optimized
- they can convert their data into [pydoxtools.Document][] and [pydoxtools.DocumentSet][]

Additionally, in order to develop a custom pipeline, pydoxtools has a large
library of [pydoxtools.operators][] which can be used to build your custom pipeline.

## Visualization of the Pipelines

The pipelines can be visualized which helps a lot when developing
your own pipeline on top of a complex pipeline such as the document pipeline.
The extraction logic for different file types can be visualized like this:

    doc = Document(fobj=make_path_absolute("./data/demo.docx"), document_type=".docx")
    # for the currently loaded file type:
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_docx.svg")
    # for the 
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_png.svg", document_logic_id=".png")

This way we can generate the pipelines for different filetypes:

- [docx](images/document_logic_docx.svg)
- [png](images/document_logic_png.svg)
  (click on links to open the images!)

Pipelines for every supported file type can be found
[here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

This also works for custom pipelines!

In order to learn more continue to: [Reference](reference)
 

