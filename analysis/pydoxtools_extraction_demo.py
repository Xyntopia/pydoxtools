# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyze the extraction of a single component from pdf

# %%
# %load_ext autoreload
# %autoreload 2

import sys

import componardo.documentx
import pydoxtools.visualization as vd

sys.path.append("../../analysis")

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# %%
import pydoxtools.extract_textstructure

# %%
from pydoxtools.extract_nlpchat import execute_task

# %%
from pydoxtools import visual_document_analysis as vda
import numpy as np
import yaml
from pydoxtools import pdf_utils, file_utils, nlp_utils, cluster_utils
import componardo.documentx
import torch
import componardo.spec_utils as su
import pathlib
import pydoxtools.cluster_utils

box_cols = pydoxtools.cluster_utils.box_cols

from tqdm import tqdm
# from IPython.display import display, Markdown, Latex

tqdm.pandas()

pdf_utils._set_log_levels()

nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %% [markdown]
# ## analyze sample html + optional pdf

# %%
training_data = pathlib.Path.home() / "comcharax/data"
# get all pdf files in subdirectory
# files = file_utils.get_nested_paths(training_data / "pdfs/datasheet", "*.pdf")
file = "/tests/data/Datasheet-Centaur-Charger-DE.6f.pdf"
files = file_utils.get_nested_paths(training_data / "sparepartsnow", "*.pdf")
len(files)

# %%
# pdf_file=random.choice(files)
file=pathlib.Path('../README.md')
file.resolve()
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
#pdf_file = training_data / "sparepartsnow/D7235-en.pdf"
print(pdf_file)

# %%
pages = np.arange(10, 15).tolist()
pages = [10, 18, 19]  # we have an unreasonable number of elements here..  what is going on?
pdf = componardo.documentx.DocumentX(pdf_file, page_numbers=pages)
x = pdf.x("elements", disk_cache=True)

# %%
# pdf.table_df0[5]

# %%
# print(page_template)

# %%
pdf.page_set

# %% [markdown]
# ## Do some layout document analysis

# %%
page = 18
pdf = componardo.documentx.DocumentX(pdf_file, page_numbers=[page])
vda.plot_box_layers(
    box_layers=[
        [pdf.line_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        [pdf.image_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False)],
        [pdf.graphic_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="yellow", filled=False)],
        [pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        # [candidate_areas, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        # [tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        # [figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        # [text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        # [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        # [t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        # [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250,
    image=pdf.images[page],
    image_box=pdf.pages_bbox[page],
),

# %% [markdown]
# access the tables as pandas dataframe!

# %%
pdf.tables_df[0]

# %% [markdown]
# ## document templates

# %%
from pydoxtools import document_base
print("".join(f"\n\n-------- {{page_{p}}} --------\n\n" + pdf.page_templates(exclude=document_base.ElementType.Table)[p] for p in pdf.pages_with_tables))

# %% [markdown]
# ## extract summary information

# %% [markdown]
# most important keywords

# %%
pdf.keywords

# %% [markdown]
# most important sentences

# %%
pdf.textrank_sents

# %%
pdf.language

# %%
pdf.page_set

# %% [markdown]
# a real summary

# %%
pdf.slow_summary

# %% [markdown]
# ## document extraction functions
#
# ### generate documentation for your pipeline

# %%
from IPython.display import Markdown

print(len(pdf.markdown_docs()))
Markdown(pdf.markdown_docs()[5000:10000])

# %% [markdown]
# ## extract a knowledge graph

# %%
#pdf.noun_graph

# %%
pdf = pydoxtools.Document(pdf_file, page_numbers=list(range(164, 166)))

# %%
#import networkx as nx
#graph = pdf.noun_graph
#graphviz = nx.nx_agraph.to_agraph(graph)
#print(str(graphviz))

# %%
svg = vd.draw(pdf.noun_graph, format='svg')
svg

# %%
pdf = pydoxtools.Document("https://en.wikipedia.org/wiki/Rocket", 
                          spacy_model_size="lg", coreference_method="fast")

# %% [markdown]
# simply get a table of all nodes in the graph

# %%
#pdf.x("graph_nodes",disk_cache=True)

# %%
pdf = pydoxtools.Document("https://en.wikipedia.org/wiki/Rocket", 
                          spacy_model_size="lg", coreference_method="fast")
KG = pdf.x("knowledge_graph",disk_cache=True)

# and visualize...
svg = vd.draw(KG, engine="fdp", format='svg')

# %%
from IPython.display import HTML, display

# %%
svg_html = '<div style="width:100%; height:100%;">{}</div>'.format(svg.data)
display(HTML(svg_html))

# %%
import networkx as nx
# Find the weakly connected components
weakly_connected_components = nx.weakly_connected_components(KG)

# Find the largest weakly connected component
sorted_weakly_connected_component = sorted(weakly_connected_components, key=len)

# %%
for i in range(1,6):
    display(vd.draw(KG.subgraph(sorted_weakly_connected_component[-i]), format="svg"))

# %%
