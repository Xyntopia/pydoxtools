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
import spacy
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

pdf = load_document(pdf_file, model_size="trf")
doc = pdf

# +
#TODO: does not work for spacy3 right now !!! import neuralcoref
#TODO: import coreferee doesn't work with "en_core_web_trf" as well!!
# -

token_props = [
    'text', 'lemma_', 'pos_', 'pos', 'tag_', 'dep_', 'dep', 'shape_',
    'head', 'subtree', 'i',
    'morph',
    'is_alpha', 'is_stop', 'is_ascii', 'is_title',
    'like_url', 'like_num', 'like_email',
    'ent_type_', 'ent_kb_id_',
    'prefix_', 'suffix_'
]
df = pd.DataFrame([[getattr(t, tp) for tp in token_props] + [t] for t in doc.spacy_doc])
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
# doc.spacy_nlp("Gas").vector

# +
# TODO: classify entities using search knn_queries? which company or organization?
# -

doc.knn_query(doc.spacy_doc[30], k=20)
doc.knn_query("modular payload", k=20)[0][0].sent

# getting similarity between entities
ents = df.loc[df.ent_type_ != ""]

e = doc.spacy_doc.ents[0]
e.start, e.end
# calculate the vector sum of every entitity

ent_vecs = np.stack([e.vector for e in doc.spacy_doc.ents])

# TODO: use umap for grouping...
ent_txt = np.array([e.text for e in doc.spacy_doc.ents])
labels, _ =cu.distance_cluster(ent_vecs, distance_threshold=0.001, pairwise_distance_func=cu.pairwise_cosine_distance)
#labels, _ = cu.distance_cluster(ent_txt, distance_threshold=0.3, pairwise_distance_func=cu.pairwise_string_diff)

labels

entdf = pd.DataFrame({"label": labels, "ents": doc.spacy_doc.ents}).groupby("label").agg(list)  #
HTML(entdf.to_html())

# +
# group labels without slow pandas groupby
# cu.merge_groups(pd.DataFrame({"label":labels, "ents":sdoc.ents}), "label")

# +
# for t in e.sent:
#    t.
# -

# tospacy_nlp-based keyembeddings# TODO: use the "document vector" + some keyword sum to get better keywords...
# with just a simple keyword this doesn't really seem to work.
# maybe also get an context-less vector by leveragin the TRF embeddings
doc.knn_query("task",filter="noun_chunks", k=20)

# +
# build noun-similarity graph

G = nx.DiGraph()
for ni,nc in enumerate(doc.noun_chunks):
    G.add_node(ni, label=nc.text, **componardo.visualization.graphviz_node_style())
    similar = doc.knn_query(nc, k=3, filter="noun_chunks", indices=True)
    #links = links[links<0.3]
    for nj,nc,d in similar:
        G.add_edge(ni, nj, weight=1-d, dir="forward")

keywords = sorted([(doc.noun_chunks[k],v) for k,v in nx.pagerank(G, weight='weight').items()], key=lambda x: x[1], reverse=True)[:20]
keywords
# -

# try document vector based kw extraction
# we check which noun-hunks are the "closest" to the document vector, assuming
# that they represent the document in the best way
doc.knn_query(doc.spacy_doc,filter="noun_chunks", k=20)

# try clustering based keyword extraction
vecs = np.stack([t.vector for t in doc.noun_chunks])
nc_txt = np.array([t.text for t in doc.noun_chunks])
labels, _ =cu.distance_cluster(vecs, distance_threshold=0.1, pairwise_distance_func=cu.pairwise_cosine_distance)
#labels, _ = cu.distance_cluster(ent_txt, distance_threshold=0.3, pairwise_distance_func=cu.pairwise_string_diff)

groups = pd.DataFrame({"label": labels, "ents": doc.noun_chunks}).groupby("label").agg(list)  #
#HTML(groups.to_html())

# +
# be careful to not install "umap" it will cause issues with umap, "umap-learn" is the right package
# pip install umap-learn pandas matplotlib datashader bokeh holoviews scikit-image and colorcet
import umap
import umap.plot

reducer = umap.UMAP(min_dist=0.001, n_neighbors=10, metric="cosine")
mapper = reducer.fit(vecs)
# -

cos_dist = cu.calc_pairwise_matrix(cu.pairwise_cosine_distance, vecs, diag=0)
cos_dist = cos_dist-cos_dist.min()
#cos_dist = cos_dist/cos_dist.max()
cos_dist.max(), cos_dist.min()

pd.DataFrame(cos_dist.flatten()).hist(bins=100)

# +
from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=30)
kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
)

vecs_pca = pca.fit(vecs).transform(vecs)
vecs_kernel_pca = kernel_pca.fit(vecs).transform(vecs)
# -

import sklearn.cluster as cluster
import hdbscan
kmeans_labels = cluster.KMeans(n_clusters=5).fit_predict(mapper.embedding_)
#cluster = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=2, metric='precomputed')
#cluster = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10, cluster_selection_epsilon=0.01)
#cluster = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=5)
cluster = hdbscan.HDBSCAN(min_cluster_size=len(doc.noun_chunks)//20)
cluster = hdbscan.HDBSCAN()
#hdbscan_labels = cluster.fit(cos_dist).labels_
hdbscan_labels = cluster.fit(mapper.embedding_).labels_
#hdbscan_labels = cluster.fit(vecs).labels_
#hdbscan_labels = cluster.fit(vecs_pca).labels_
print(hdbscan_labels)
df_labels = pd.DataFrame({"label": hdbscan_labels, "txt": nc_txt, "nc":list(range(len(nc_txt)))})
df_labels = pd.DataFrame({"label": kmeans_labels, "txt": nc_txt, "nc":list(range(len(nc_txt)))})
groups = df_labels.groupby("label").agg(list)  #
HTML(groups.to_html())

kw = []
for i in groups.index:
    if i>-1:
        group_vec = pd.Series(doc.noun_chunks)[groups.nc[i]].apply(lambda x: x.vector).mean()
        res = doc.knn_query(group_vec, k=2, filter="noun_chunks")
        kw.append((i,*res[:]))
kw

umap.plot.points(mapper, labels=hdbscan_labels, show_legend=False)

umap.plot.points(mapper, labels=kmeans_labels, show_legend=False)

#p = umap.plot.interactive(mapper, labels=fmnist.target[:30000], hover_data=hover_data, point_size=2)
#hover_data = {ni:nc.text for ni,nc in doc.noun_chunks.items()}
p = umap.plot.interactive(mapper, point_size=10, labels=hdbscan_labels, hover_data=df_labels)
umap.plot.show(p)


