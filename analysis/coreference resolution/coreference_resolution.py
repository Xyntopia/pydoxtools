# # Play around with rule-based coreference resolution strategies

# +

# %load_ext autoreload
# %autoreload 2

import logging
from pathlib import Path

import pandas as pd
import spacy
import torch
from tqdm import tqdm

from pydoxtools import nlp_utils, Document
from pydoxtools import pdf_utils, file_utils
from pydoxtools.settings import settings

import pydoxtools.visualization as vd
import html

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

tqdm.pandas()

pdf_utils._set_log_levels()

nlp_utils.device, torch.cuda.is_available(), torch.__version__
# -

# ## analyze sample html + optional pdf

files = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
len(files)

pdf_file = settings.TRAINING_DATA_DIR / "pdfs/whitepaper/En-Sci-Application-Brief.88.pdf"
# pdf_file=random.choice(files)
print(pdf_file.absolute())

# get all pdf files in subdirectory
training_data_dir = 0
pdf_file = Path("../../README.md")

pdf = Document(pdf_file, spacy_model_size="sm")
doc = pdf

doc.configuration

relationships=doc.relationships
relationships

text=doc.spacy_doc.text
G = doc.knowledge_graph
#G=KG

vd.draw(G, format='svg')

# +
#dotgraph.write()

# +
#print(str(dotgraph)[:1000])
# -

if False:
    line = 1429
    context = 10
    for i,l in enumerate(str(dotgraph).split("\n")[line-context:line+context]):
        print(f"{i+line-context}: {l}")

from fastcoref import FCoref
from fastcoref import LingMessCoref

model = FCoref()
model = LingMessCoref()

preds = model.predict(
   texts=[[str(t) for t in doc.spacy_doc]],
   is_split_into_words=True
)

tok_id_coreferences = preds[0].get_clusters(as_strings=False)
tok_id_coreferences

tok_id_coreferences

# +
#G[doc.spacy_doc[4]]
for cr_group in tok_id_coreferences:
    # check if any node in our group is represented in our relationship graph
    nodes = [t for t in cr_group if G.has_node(t[0])]
            
    # merge nodes
    if nodes:
        
        
        
    
        
#nG = nx.identified_nodes(G, node,node2)
# -

import networkx as nx
node=106
connected_nodes = list(next(comp for comp in nx.weakly_connected_components(nG) if node in comp))
subgraph = nG.subgraph(connected_nodes)
vd.draw(subgraph, format="svg")

preds[0].get_clusters()

preds[0].get_logit(
   span_i=(106, 107), span_j=(138, 139)
)


