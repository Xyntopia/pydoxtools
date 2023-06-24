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
# # Analyze the extraction of tables from pdfs

# %%
import sys

import pydoxtools.extract_tables

sys.path.append("..")
# %load_ext autoreload
# %autoreload 2

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# %%
from pydoxtools import nlp_utils
from pydoxtools import pdf_utils
from pydoxtools import visual_document_analysis as vda
from pydoxtools import cluster_utils as gu
import numpy as np
import pandas as pd

import pathlib
box_cols = gu.box_cols
from pydoxtools.settings import settings
import torch
from IPython.display import display, HTML

from tqdm import tqdm

tqdm.pandas()

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %%
x0,y0,x1,y1 = 1,2,3,4

# %% [markdown]
# ## first of all, get some correctly identified areas

# %%
# #!pip install datasets[audio]

# %%
#from datasets import load_dataset

#dataset = load_dataset("bsmock/pubtables-1m")

# %%
training_data = pathlib.Path.home() / "comcharax/data"
page =19 # we have an unreasonable number of elements here..  what is going on?
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %%
pdf=pydoxtools.Document(pdf_file, page_numbers=[page])
img = pdf.images[page]

# %% [markdown]
# ## calculate table text for the page

# %%
pdf.table_candidates

# %%
t=pdf.table_candidates[1]

# %%
pretty_print(t.df)
#t.df
#pdfi.pages[page]
#t.is_valid

# %%
#pdfi.table_metrics
#pdfi.table_metrics_X

# %%
#[pretty_print(t) for t in pdfi.tables_df]

# %%
#pdfi.tables_df[0]

# %% [markdown]
# ## text-only extraction

# %%
# detect vertical lines for column detection
# we are using lines here, but it might be better in the future to use
# words for this  if some lines get detected "wrongly"

le=t.df_words.copy()#[box_cols]

# then check which other cells are overlapping with that cell...
# we first calculate the overlap distance for rows & columns
# and then use that to group them into columns & rows
le_col_overlap = gu.calc_pairwise_matrix(
    gu.pairwise_box_gap_distance_along_axis_func, le[box_cols].values, diag=-1, axis=0)
le_row_overlap = gu.calc_pairwise_matrix(
    gu.pairwise_box_gap_distance_along_axis_func, le[box_cols].values, diag=-1, axis=1
)
# create row & column clusters with the overlap distance matrix
# all negative distances mean that two cells overlap, so we set the thresold to 0
le["cols"],_=gu.distance_cluster(data = le[box_cols].values,
                 distance_matrix=le_col_overlap, distance_threshold=0)
le["rows"],_=gu.distance_cluster(data = le[box_cols].values,
                 distance_matrix=le_row_overlap, distance_threshold=0)
cols = gu.merge_bbox_groups(le, "cols").sort_values(by="x0", ascending=True)
rows = gu.merge_bbox_groups(le, "rows").sort_values(by="y0", ascending=False)

# create new column-coordinates in the correct order..
cols=cols.reset_index()
rows=rows.reset_index()

# and map the sorted column & row coordinates to the text boxes
col_map = pd.Series(cols.index.values, index=cols['index'] )
row_map = pd.Series(rows.index.values, index=rows['index'] )
le["cols"]=le["cols"].map(col_map)
le["rows"]=le["rows"].map(row_map)
le = le.sort_values(by=["cols","rows"])

# now all we need to do is use the row & column coordinates to create a dataframe!
table = pd.pivot_table(le, values='text',
                        index=['rows'], columns=['cols'],
                        aggfunc='first', fill_value='')
table

# %%
margin=10
layers = [
    #[t.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.05, color="blue", filled=True)],
    #[t.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.2, color="orange")],
    [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.4, color="orange")],
    [t.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
    [cols[vda.box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False, box_numbers=True)],
    [rows[vda.box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False, box_numbers=True)],
]

vda.plot_box_layers(
    box_layers=layers,
    bbox=t.bbox+[-margin,-margin,margin,margin], dpi=250
),

# %% [markdown]
# ## plot table candidate

# %%
margin=20
# make cells "smaller" than in reality to make borders between the more obvious
borders =b= 2 
#ca = t.detect_cells()
ca = t.detect_cells(steps=None).copy()
layers = [
    [t.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.05, color="blue", filled=True)],
    #[t.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.2, color="orange")],
    [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.4, color="orange")],
]
oc = t._debug.get("open_cells", pd.DataFrame())
if not oc.empty:
    oc = oc[vda.box_cols].values + [b,b,-b,-b]
    layers.append([oc, vda.LayerProps(alpha=0.2, color="black")])

if not ca.empty:
    layers += [[ca[box_cols].values + [b,b,-b,-b], vda.LayerProps(alpha=0.2, color="random")]]
vda.plot_box_layers(
    box_layers=layers,
    bbox=t.bbox+[-margin,-margin,margin,margin], dpi=250
),

# %%
#ca

# %%
oc

# %% [markdown]
# ## plot table candidate areas

# %%
# vda.plot_boxes(
#    dfl[vda.box_cols].values,
#    groups = dfl["hm"].values,
#    bbox = None, dpi=250)#p.page_bbox)

print(pdf_file)
table_candidates, box_iterations = pdf.table_candidates, pdf.table_box_levels

# %%
ERROR = np.array([[199.79,  522.362, 291.686, 579.698]])

vda.plot_box_layers(
    box_layers=[
        [pdf.graphic_elements[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red", filled=False)],
        [pdf.line_elements[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        [box_iterations[page][0].values[:,:4], vda.LayerProps(alpha=0.5, color="black", filled=False)],
        [box_iterations[page][1].values[:,:4], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        #[pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[boxes.values, vda.LayerProps(alpha=0.1, color="blue", filled=False)],
        #[p.table_candidate_boxes_df.values, vda.LayerProps(
        #    alpha=1.0, color="red", filled=False, box_numbers=True)]
        #[ERROR,vda.LayerProps(color="red")]
    ],
    bbox=pdf.pages_bbox[page], dpi=250
),

# %%
