# # Extract keywords from documents

# +

# %load_ext autoreload
# %autoreload 2

import logging
from operator import attrgetter

import numpy as np
import pandas as pd
import torch
import spacy
import networkx as nx
from componardo import visualization as vz
from IPython.display import HTML
from pydoxtools import nlp_utils, load_document
from pydoxtools import pdf_utils, file_utils, cluster_utils as cu
from pydoxtools.settings import settings
from tqdm import tqdm

# be careful to not install "umap" it will cause issues with umap, "umap-learn" is the right package
# pip install umap-learn pandas matplotlib datashader bokeh holoviews scikit-image and colorcet
import umap.plot

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

import pytextrank
pdf = load_document(pdf_file, model_size="trf")
pdf.spacy_nlp.add_pipe("textrank")
doc = pdf

# +
#TODO: does not work for spacy3 right now !!! import neuralcoref
#TODO: import coreferee doesn't work with "en_core_web_trf" as well!!

# +
# df.to_dict('index').items()

# +
# doc.spacy_nlp("Gas").vector

# +
# TODO: classify entities using search knn_queries? which company or organization?
# -

doc.knn_query(doc.spacy_doc[30], k=20)
doc.knn_query("modular payload", k=20)[0][0].sent

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

G = doc.similarity_graph(k=4, max_distance=0.2, method="noun_chunks")
G = vz.simple_graphviz_styling(G)
#vz.draw(G, "neato")

for phrase in doc.spacy_doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)

# build noun-similarity graph
doc.textrank_keywords(k=10, max_links=3, max_distance=0.2, method="noun_chunks")

# build noun-similarity graph
doc.textrank_keywords(k=3, max_distance=0.2)

# try document vector based kw extraction
# we check which noun-hunks are the "closest" to the document vector, assuming
# that they represent the document in the best way
doc.knn_query(doc.spacy_doc,filter="noun_chunks", k=30)

# try clustering based keyword extraction
vecs = np.stack([t.vector for t in doc.noun_chunks])
nc_txt = np.array([t.text for t in doc.noun_chunks])
labels, _ =cu.distance_cluster(vecs, distance_threshold=0.1, pairwise_distance_func=cu.pairwise_cosine_distance)
#labels, _ = cu.distance_cluster(ent_txt, distance_threshold=0.3, pairwise_distance_func=cu.pairwise_string_diff)

groups = pd.DataFrame({"label": labels, "ents": doc.noun_chunks}).groupby("label").agg(list)  #
#HTML(groups.to_html())

# %%time
reducer = umap.UMAP(
    n_neighbors=20, #determines "locality" of the clusters
    min_dist=0.0, #packing distance 0 for clustering to get better clusters ;)
    n_components=5, #final dimensionality 
    random_state=42,
    metric="cosine"
)
mapper = reducer.fit(vecs)

cos_dist = cu.calc_pairwise_matrix(cu.pairwise_cosine_distance, vecs, diag=0)
cos_dist = cos_dist-cos_dist.min()
#cos_dist = cos_dist/cos_dist.max()
cos_dist.max(), cos_dist.min()

pd.DataFrame(cos_dist.flatten()).hist(bins=100)

import sklearn.cluster as cluster
import hdbscan
kmeans_labels = cluster.KMeans(n_clusters=5).fit_predict(mapper.embedding_)
# n_init iterations the algorithm is run
spectral_labels = cluster.SpectralClustering(n_clusters=5, random_state=42, n_components=5, n_init=10).fit_predict(mapper.embedding_)
#cluster = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=2, metric='precomputed')
#cluster = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10, cluster_selection_epsilon=0.01)
#cluster = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=5)
cluster = hdbscan.HDBSCAN(min_cluster_size=len(doc.noun_chunks)//35)
#cluster = hdbscan.HDBSCAN()
#hdbscan_labels = cluster.fit(cos_dist).labels_
hdbscan_labels = cluster.fit(mapper.embedding_).labels_
#hdbscan_labels = cluster.fit(vecs).labels_
#hdbscan_labels = cluster.fit(vecs_pca).labels_
print(hdbscan_labels)
labels=spectral_labels
df_labels = pd.DataFrame({"label": labels, "txt": nc_txt, "nc":list(range(len(nc_txt)))})
groups = df_labels.groupby("label").agg(list)  #
HTML(groups.to_html())

kw = []
for i in groups.index:
    if i>-1:
        group_vec = pd.Series(doc.noun_chunks)[groups.nc[i]].apply(lambda x: x.vector).mean()
        res = doc.knn_query(group_vec, k=2, filter="noun_chunks")
        kw.append((i,*res[:]))
kw

mapper2D = umap.UMAP(
    n_neighbors=10, #determines "locality" of the clusters
    min_dist=0.0, #packing distance 0 for clustering to get better clusters ;)
    n_components=2, #final dimensionality 
    random_state=0,
    metric="cosine"
).fit(vecs)
umap.plot.points(mapper2D, labels=labels, show_legend=False)
#umap.plot.points(mapper2D, labels=kmeans_labels, show_legend=False)

#p = umap.plot.interactive(mapper, labels=fmnist.target[:30000], hover_data=hover_data, point_size=2)
#hover_data = {ni:nc.text for ni,nc in doc.noun_chunks.items()}
p = umap.plot.interactive(mapper2D, point_size=10, labels=labels, hover_data=df_labels)
umap.plot.show(p)


