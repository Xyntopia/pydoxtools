# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import pydoxtools as pdx
import pydoxtools.visualization as vd
from IPython.display import SVG, display, HTML, Image

doc = pdx.Document(Path("../../pydoxtools/README.md"), 
                          spacy_model_size="sm", coreference_method="fast", 
                   graph_debug_context_size=0)

doc.elements
