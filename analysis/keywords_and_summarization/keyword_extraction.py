# # Extract keywords from documents

# +

# %load_ext autoreload
# %autoreload 2

import logging
from operator import attrgetter

import componardo.visualization
import hnswlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from spacy.language import Language
from spacy.tokens import Doc
from IPython.core.display import display
from IPython.display import HTML
from pydoxtools import nlp_utils, load_document
from pydoxtools import pdf_utils, file_utils, cluster_utils as cu
from pydoxtools.settings import settings
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__
# -

# ## analyze sample html + optional pdf

# +
# get all pdf files in subdirectory

files = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
len(files)
# -

pdf_file = settings.TRAINING_DATA_DIR / "pdfs/whitepaper/En-Sci-Application-Brief.88.pdf"
# pdf_file=random.choice(files)
print(pdf_file.absolute())

pdf = load_document(pdf_file)
doc = pdf

# +
#TODO: does not work for spacy3 right now !!! import neuralcoref
#TODO: import coreferee doesn't work with "en_core_web_trf" as well!!
# -

nlpmodelid = nlp_utils.get_spacy_model_id('de', 'medium')
# nlpmodelid = "de_core_news_lg" #sm, md, lg and "trf" for transformer
# nlpmodelid = "de_dep_news_trf"
nlpmodelid = "en_core_web_trf"
# nlpmodelid = "en_core_web_lg"
nlpmodelid
nlp = nlp_utils.load_cached_spacy_model(nlpmodelid)

nlp.add_pipe('trf_vectors')
# nlp.add_pipe('coreferee')
# nlp.add_pipe("merge_entities") #TODO: we can not do this, as we'll get the wrong BERT vector mapping to tokens... :(
# nlp.add_pipe("merge_noun_chunks") #TODO: we can not do this, as we'll get the wrong BERT vector mapping to tokens... :(
# sdoc = nlp(doc.full_text[:1000])
sdoc = nlp(doc.full_text)

sdoc.has_annotation("DEP")

# +
# list(sdoc.noun_chunks)
# -

spacy.explain('CD')

t = sdoc[27]
t.tag_

token_props = [
    'text', 'lemma_', 'pos_', 'pos', 'tag_', 'dep_', 'dep', 'shape_',
    'head', 'subtree', 'i',
    'morph',
    'is_alpha', 'is_stop', 'is_ascii', 'is_title',
    'like_url', 'like_num', 'like_email',
    'ent_type_', 'ent_kb_id_',
    'prefix_', 'suffix_'
]
df = pd.DataFrame([[getattr(t, tp) for tp in token_props] + [t] for t in sdoc])
df.columns = token_props + ['tok']
# df.query('ent_type_!=""')
sel = ['i', 'text', 'pos_', 'pos', 'tag_', 'dep_', 'head', 'ent_type_', 'ent_kb_id_']
df = df[sel]
df['head.pos_'] = df['head'].apply(attrgetter('pos_'))
df['head.dep_'] = df['head'].apply(attrgetter('dep_'))
df['head.i'] = df['head'].apply(attrgetter('i'))
df.dep_ = df.dep_.apply(spacy.explain)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# display(df.head(20))
# df[sel]
df.query('ent_type_!=""')

# +
# df.to_dict('index').items()

# +
# nx.write_graphml_lxml(G,'test.graphml')
# nx.write_graphml(G,'test.graphml')

# +
# model,tokenizer = nlp_utils.load_models()
# v = nlp_utils.longtxt_embeddings_fullword(doc.full_text, model, tokenizer)
# v[1]
# pd.DataFrame(list(zip(v[1],df['text'])), columns=['trf','tok']).head(50)

# x = pd.DataFrame(list(zip(v[1],df.query("pos!=103")['text'])), columns=['trf','tok'])
# x

# x.query("trf!=tok")

# df.query("pos!=103")['text']

# +
p = hnswlib.Index(space='cosine', dim=sdoc._.trf_token_vecs.shape[1])
# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=len(sdoc) + 1, ef_construction=200, M=16)

# Element insertion (can be called several times):
p.add_items(data=sdoc._.trf_token_vecs, ids=df.index)
# Controlling the recall by setting ef:
p.set_ef(100)  # ef should always be > k
# -

similar = p.knn_query([sdoc[25].vector], k=20)
display(similar[1].round(3))
# sdoc[similar[0][0]]
similar[0][0]

spacy.explain("NNP")

# getting similarity between entities
ents = df.loc[df.ent_type_ != ""]

e = sdoc.ents[0]
e.start, e.end
# calculate the vector sum of every entitity

ent_vecs = np.stack([vecs[e.start: e.end].mean(0) for e in sdoc.ents])

ent_vecs

ent_txt = np.array([e.text for e in sdoc.ents])
# labels, _ =cu.distance_cluster(ent_vecs, distance_threshold=0.05, pairwise_distance_func=pairwise_cosine_distance)
labels, _ = cu.distance_cluster(ent_txt, distance_threshold=0.3, pairwise_distance_func=cu.pairwise_string_diff)

labels

entdf = pd.DataFrame({"label": labels, "ents": sdoc.ents}).groupby("label").agg(list)  #
HTML(entdf.to_html())

# +
# group labels without slow pandas groupby
# cu.merge_groups(pd.DataFrame({"label":labels, "ents":sdoc.ents}), "label")

# +
# for t in e.sent:
#    t.
# -

words = list(sdoc.noun_chunks)

# +
noun_vecs = np.stack([vecs[e.start: e.end].mean(0) for e in words])
noun_ids = {i:nc for i,nc in enumerate(words)}

p = hnswlib.Index(space='cosine', dim=vecs.shape[1])
# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=len(noun_vecs) + 1, ef_construction=200, M=16)

# Element insertion (can be called several times):
p.add_items(data=noun_vecs, ids=list(noun_ids.keys()))
# Controlling the recall by setting ef:
p.set_ef(100)  # ef should always be > k

#similar = p.knn_query([vecs[25]], k=20)
#display(similar[1].round(3))
# sdoc[similar[0][0]]
#similar[0][0]

# build noun-similarity graph

G = nx.DiGraph()
for ni,nt in noun_ids.items():
    G.add_node(ni, label=nt, **componardo.visualization.graphviz_node_style())
    similar = p.knn_query([noun_vecs[ni]], k=3)
    links = similar[1][0,1:]
    #links = links[links<0.3]
    for j,w in zip(similar[0][0,1:], links):
        G.add_edge(ni, j, weight=1-w, dir="forward")

keywords = sorted([(noun_ids[k],v) for k,v in nx.pagerank(G, weight='weight').items()], key=lambda x: x[1], reverse=True)[:20]
keywords

# +
#display(componardo.visualization.draw(G))

# +
# try out coreference based on nearest neighbours..

list(sdoc.sents)[10]

# +
# which company or organization?
# -

search_word = nlp("company")
search_word.

similar = p.knn_query([noun_vecs[ni]], k=3)


