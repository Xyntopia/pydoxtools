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

# +
import networkx as nx

relationships=doc.relationships
text=doc.spacy_doc.text

def build_knowledge_graph(relationships, text):
    KG = nx.DiGraph()

    ct_size = 100
    for rel_type, rel_list in relationships.items():
        for tok in rel_list:
            t1,t2 = tok[0], tok[-1]
            n1,n2 = t1.idx, t2.idx
            tx1, tx2 = t1.text, t2.text
            tx1, tx2 = html.escape(tx1),html.escape(tx2)
            #tx1, tx2 = tx1.replace('\\n','\n'), tx2.replace('\\n','\n')
            tx1, tx2 = tx1.replace('\\','\\\\'), tx2.replace('\\','\\\\')
            ct1 = text[max(0,n1-ct_size):n1+ct_size]
            ct1 = ct1[:ct_size]+"||"+ct1[ct_size:len(tx1)+ct_size]+"||"+ct1[len(tx1)+ct_size:]
            ct2 = text[max(0,n2-ct_size):n2+ct_size]
            ct2 = ct2[:ct_size]+"||"+ct2[ct_size:len(tx2)+ct_size]+"||"+ct2[len(tx2)+ct_size:]
            #ct1 = ct1.replace('\\n','\n')
            #ct1 = ct1.replace("\\","\\\\").replace("<","\<").replace(">","\>")
            #tx1+=f"{{{ct1}}}"
            #tx1 =f"{tx1}" + "\n\n" + ct1
            ct1 = html.escape(ct1).replace('\n', '<br/>')
            tx1 = f'<<font point-size="15">{tx1}</font><br/><font point-size="8">{ct1}</font>>'
            ct2 = html.escape(ct2).replace('\n', '<br/>')
            tx2 = f'<<font point-size="15">{tx2}</font><br/><font point-size="8">{ct2}</font>>'
            KG.add_node(n1, label=fr'{tx1}', shape="box")
            KG.add_node(n2, label=fr'{tx2}', shape="box")
            if rel_type == 'SVO':
                label=tok[1].text.replace('\\','\\\\')
            elif rel_type in ['Attribute', 'Adjective']:
                label='is'
            elif rel_type == 'Prepositional':
                label='related'
            elif rel_type == 'Possessive':
                label='owns'
            KG.add_edge(n1,n2, label=label)

    return KG

G=build_knowledge_graph(relationships, doc.spacy_doc.text)
#G=KG

# +
#rel_list = relationships['SVO']
#[(rel[0].text, rel[2].text, {'type': rel[1].text}) for rel in rel_list]

# +
#list(KG.edges.items())

# +
#list(G.nodes.items())
# -

from IPython.core.display import SVG
dotgraph = nx.nx_agraph.to_agraph(G)
dotgraph.graph_attr["overlap"] = "false"
svg=dotgraph.draw(prog='dot', format='svg')

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

SVG(svg)




