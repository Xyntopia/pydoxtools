# # Analyze the extraction of a single component from pdf

# +
import sys

# %load_ext autoreload
# %autoreload 2

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# +
from pathlib import Path

from pydoxtools import nlp_utils, load_document
from pydoxtools import pdf_utils, file_utils, cluster_utils as cu
from pydoxtools.settings import settings
import torch
import hnswlib
import numpy as np
import random
from operator import attrgetter

import networkx as nx
import pandas as pd
from tqdm import tqdm
from IPython.display import display, Markdown, Latex
import os
from os.path import join

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__

TRAINING_DIR = Path("../../../training_data/")
# -

# ## analyze sample html + optional pdf

# +
# get all pdf files in subdirectory

files = file_utils.get_all_files_in_nested_subdirs(TRAINING_DIR / "pdfs/datasheet", "*.pdf")
len(files)
# -

pdf_file = TRAINING_DIR / "pdfs/terms/termsofuse-americas.34.pdf"
#pdf_file=random.choice(files)
print(pdf_file.absolute())

pdf = load_document(pdf_file)
pdf

# d = pdf_utils.extract_pdf_data(str(pdf_file))
with open(pdf_file, "rb") as pf:
    doc = load_document(pf, source="jupyter_lab")
    d = doc.model

len(d.tables)
d.tables[0]

doc.urls

txt = doc.full_text
#doc.textboxes
#txt

# +
#nlp_utils.init_nlp_cache()

# +
#print(txt)

# +
# #!python -m spacy download xx_ent_wiki_sm

# +
#nlp_utils.download_nlp_models()
# -

nlpmodelid = nlp_utils.get_spacy_model_id('de', 'medium')
#nlpmodelid = "de_core_news_lg" #sm, md, lg and "trf" for transformer
#nlpmodelid = "de_dep_news_trf"
nlpmodelid = "en_core_web_trf"
nlpmodelid

nlp = nlp_utils.load_cached_spacy_model(nlpmodelid)
#sdoc = nlp(doc.full_text[:3000])
sdoc = nlp(doc.full_text)

# +
#sdoc
# -

sdoc.has_annotation("DEP")

# +
#list(sdoc.noun_chunks)
# -

import spacy
spacy.explain('_SP')

t = sdoc[27]
t.tag_

token_props = [
    'text','lemma_','pos_','pos','tag_','dep_','dep','shape_',
    'head','subtree','i',
    'morph',
    'is_alpha','is_stop', 'is_ascii','is_title',
    'like_url','like_num','like_email',
    'ent_type_','ent_kb_id_',
    'prefix_','suffix_'
]
df = pd.DataFrame([[getattr(t, tp) for tp in token_props]+[t] for t in sdoc])
df.columns=token_props+['tok']
#df.query('ent_type_!=""')
sel = ['i','text','pos_','pos','tag_','dep_','head','ent_type_','ent_kb_id_']
df = df[sel]
df['head.pos_']=df['head'].apply(attrgetter('pos_'))
df['head.dep_']=df['head'].apply(attrgetter('dep_'))
df['head.i']=df['head'].apply(attrgetter('i'))
df.dep_ = df.dep_.apply(spacy.explain)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#display(df.head(20))
    #df[sel]
df.query('ent_type_!=""')

# +
#df.to_dict('index').items()

# +
import pyyed

def create_example_graph():
    g = pyyed.Graph()

    g.add_node('foo', font_family="Zapfino")
    g.add_node('foo2', shape="roundrectangle", font_style="bolditalic", underlined_text="true")

    g.add_edge('foo1', 'foo2')
    g.add_node('abc', font_size="72", height="100", shape_fill="#FFFFFF")

    g.add_node('bar', label="Multi\nline\ntext")
    g.add_node('foobar', label="""Multi
        Line
        Text!""")

    g.add_edge('foo', 'foo1', label="EDGE!", width="3.0", color="#0000FF",
                   arrowhead="white_diamond", arrowfoot="standard", line_type="dotted")

    graphstring = g.get_graph()
    return graphstring

def save_to_file(g):
    # To write to file:
    with open('test_graph.graphml', 'w') as fp:
        fp.write(g.get_graph())

    # Or:
    #g.write_graph('example.graphml')

    # Or, to pretty-print with whitespace:
    #g.write_graph('pretty_example.graphml', pretty_print=True)


# +
# build the graph
#G = nx.Graph()
#G.add_nodes_from(
#    df.to_dict('index').items()
#)
# -

# Valid node shapes are: "rectangle", "rectangle3d", "roundrectangle", "diamond", "ellipse", "fatarrow", "fatarrow2", "hexagon", "octagon", "parallelogram", "parallelogram2", "star5", "star6", "star6", "star8", "trapezoid", "trapezoid2", "triangle", "trapezoid2", "triangle"
#
# Valid line_types are: "line", "dashed", "dotted", "dashed_dotted"
#
# Valid font_styles are: "plain", "bold", "italic", "bolditalic"
#
# Valid arrow_types are: "none", "standard", "white_delta", "diamond", "white_diamond", "short", "plain", "concave", "concave", "convex", "circle", "transparent_circle", "dash", "skewed_dash", "t_shape", "crows_foot_one_mandatory", "crows_foot_many_mandatory", "crows_foot_many_optional", "crows_foot_many_optional", "crows_foot_one", "crows_foot_many", "crows_foot_optional"

# +
g = pyyed.Graph()
for tok in sdoc:
    color = "#ffcc00"
    if tok.pos_=="NOUN":
        color = "#00ccff"
    elif tok.pos_ =="VERB":
        color= "#00ff00"
    g.add_node(tok.i, label=tok.text, shape_fill=color)
    
for tok in sdoc:
    if not tok.head.i==tok.i:
        g.add_edge(tok.head.i, tok.i, label="EDGE!", width="1.0", color="#000000", 
               arrowhead="standard", arrowfoot="none", line_type="line")

save_to_file(g)

# +
#nx.write_graphml_lxml(G,'test.graphml')
#nx.write_graphml(G,'test.graphml')

# +
#model,tokenizer = nlp_utils.load_models()
#v = nlp_utils.longtxt_embeddings_fullword(doc.full_text, model, tokenizer)
#v[1]
#pd.DataFrame(list(zip(v[1],df['text'])), columns=['trf','tok']).head(50)

#x = pd.DataFrame(list(zip(v[1],df.query("pos!=103")['text'])), columns=['trf','tok'])
#x

#x.query("trf!=tok")

#df.query("pos!=103")['text']
# -

# where are the spacy token boundaries in the trf vectors
vec_idx_splits = np.cumsum(sdoc._.trf_data.align.lengths)
# get transformer vectors
trf_vecs = sdoc._.trf_data.tensors[0].reshape(-1,768)
# calculate mapping groups from spacy tokens to transformer vector indices
vec_idxs = np.split(sdoc._.trf_data.align.dataXd,vec_idx_splits)

# take sum of mapped transformer vector indices for spacy vectors
vecs=np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])

vecs.shape

p = hnswlib.Index(space='cosine', dim=vecs.shape[1])
# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=len(vecs)+1, ef_construction=200, M=16)

# Element insertion (can be called several times):
p.add_items(data=vecs, ids=df.index)
# Controlling the recall by setting ef:
p.set_ef(100)  # ef should always be > k

similar = p.knn_query([vecs[25]], k=20)
display(similar[1].round(3))
#sdoc[similar[0][0]]
similar[0][0]

[sdoc[i] for i in similar[0][0]]

spacy.explain("NNP")

# getting similarity between entities
ents = df.loc[df.ent_type_!=""]

e = sdoc.ents[0]
e.start, e.end
# calculate the vector sum of every entitity

ent_vec = vecs[e.start: e.end].mean(0)

sdoc[e.start:e.end]

# +
# cluster entities
