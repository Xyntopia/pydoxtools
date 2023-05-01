# Welcome to Pydoxtools Documentation!

[Readme](readme_cp)

## Introduction

The main interface that pydoxtools uses are three classes:

- [pydoxtools.Document][]
- [pydoxtools.DocumentSet][]
- [pydoxtools.Pipeline][]

And a set of operators:

- [pydoxtools.operators][]

### Analyzing Document

Document & DocumentSet are both using [pydoxtools.Pipeline][] and
predefine a complex pipeline to extract data from individual documents or
a set of documents. A list of all the "out-of-the-box" featurs for each pipeline
can be found in:

-> [Pipelines](pipelines)

### Building your own Pipelines with LLMs (Large Language Models) and other types of AI

the Pipeline class can be used to build complex, custom pipelines which
have several out-of-the-box features which makes them easy to use in
modern pipelines involving the use of a lot of AI tools:

- they can be mixed, extended, (partially) overwritten with other pipelines
- they can export/import their data (yaml, json, python-dict),
- they can be configured & optimized
- they can convert their data into [pydoxtools.Document][] and [pydoxtools.DocumentSet][]

Additionally, in order to develop a custom pipeline, pydoxtools has a large
library of [pydoxtools.operators][] which can be used to build your custom pipeline.
It usually makes sense to use [pydoxtools.Document][] or [pydoxtools.DocumentSet][]
as a base for a new pipeline and only replace small parts of them in order to
get desired custom functionality.

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
 

