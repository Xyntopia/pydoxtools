# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import pydoxtools as pdx
import pydoxtools.visualization as vd
from IPython.display import SVG, display, HTML, Image

if True:
    pdf = pdx.Document("https://en.wikipedia.org/wiki/Rocket", 
                              spacy_model_size="sm", coreference_method="fast", graph_debug_context_size=0)
    KG = pdf.x("document_graph")

    jpg = vd.draw(KG, engine="fdp", format='jpg')
Image(jpg)

pdf = pdx.Document(Path("../../pydoxtools/README.md"), 
                          spacy_model_size="sm", coreference_method="fast", 
                   graph_debug_context_size=0)
KG = pdf.x("document_graph")

jpg = vd.draw(KG, engine="fdp", format='jpg')
Image(jpg)

# and visualize...
svg = vd.draw(KG, engine="fdp", format='graphviz')

print(svg)


