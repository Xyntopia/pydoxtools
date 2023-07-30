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

t = doc.spacy_doc[90]
doc.full_text[t.idx:t.idx+len(t)]

# +
import networkx as nx

relationships=doc.relationships

#def build_knowledge_graph(relationships):
KG = nx.Graph()

for rel_type, rel_list in relationships.items():
    for tok in rel_list:
        t1,t2 = tok[0], tok[-1]
        n1,n2 = t1.idx, t2.idx
        tx1, tx2 = t1.text.replace('\\','\\\\'), t2.text.replace('\\','\\\\')
        KG.add_node(n1, label=fr'{tx1}')#, shape="ellipse")
        KG.add_node(n2, label=fr'{tx2}', shape="ellipse")
        if rel_type == 'SVO':
            label=tok[1].7	[label="\",
543: 		shape=ellipstext.replace('\\','\\\\')
        elif rel_type in ['Attribute', 'Adjective']:
            label='is'
        elif rel_type == 'Prepositional':
            label='related'
        elif rel_type == 'Possessive':
            label='owns'
        KG.add_edge(n1,n2, label=label)

    #return KG

import pydoxtools.visualization as vd
G=KG
#G=build_knowledge_graph(relationships)

# +
#rel_list = relationships['SVO']
#[(rel[0].text, rel[2].text, {'type': rel[1].text}) for rel in rel_list]

# +
#list(KG.edges.items())

# +
#list(G.nodes.items())
# -

from IPython.core.display import SVG
graph=G
dotgraph = nx.nx_agraph.to_agraph(graph)
dotgraph.graph_attr["overlap"] = "false"
svg=dotgraph.draw(prog='dot', format='svg')

# +
#for i,l in enumerate(str(dotgraph).split("\n")):
#    print(f"{i}: {l}")
# -

SVG(svg)
