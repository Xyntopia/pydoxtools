# Welcome to Pydoxtools Documentation!

## Readme

[Readme](readme_cp)

## Visualization of the Extraction logic

The extraction logic for different file types can be visualized bydoing something like this:

    doc = Document(fobj=make_path_absolute("./data/demo.docx"), document_type=".docx")
    # for the currently loaded file type:
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_docx.svg")
    # for the 
    doc.logic_graph(image_path=settings._PYDOXTOOLS_DIR / "docs/images/document_logic_png.svg", document_logic_id=".png")

This way we can generate the pipelines for different filetypes:

- [docx](images/document_logic_docx.svg)
- [png](images/document_logic_png.svg)

(click on links to open the images!)

This also works for custom pipelines!

In order to learn more continue to: [Reference](reference)
 

