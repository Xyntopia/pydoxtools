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

import pydoxtools.visual_document_analysis

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
from pydoxtools import cluster_utils
box_cols = cluster_utils.box_cols
from pydoxtools.settings import settings
import torch
from IPython.display import display
import pdfminer
import pdfminer.layout
import pathlib
import sklearn
import sklearn.cluster as cluster

import numpy as np

import pandas as pd
from tqdm import tqdm
import os
from os.path import join
from IPython.display import display, HTML

tqdm.pandas()

pdf_utils._set_log_levels()

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))

nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %% [markdown]
# ## load pdf files

# %%
# get all pdf files in subdirectory
files = [join(root, f) for root, dirs, files
         in os.walk(settings.PDFDIR)
         for f in files if f.endswith(".pdf")]
len(files)

# %% [markdown]
# ## try to extract table boundaries using our own method
#
# 1. first we will group lines into boxes to identify "whole" cells the grouping was already provided by pdfminer.six --> **box_groups**
# 2. afterwards we will group those boxes into columns & rows --> **valid_columns**
# 3. as the third step, cluster the columns into tables --> table_groups

# %%
training_data = pathlib.Path.home() / "comcharax/data"
page =19 # we have an unreasonable number of elements here..  what is going on?
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %%
img = pydoxtools.Document(pdf_file, page_numbers=[page]).images[page]

# %%
#pdf = pydoxtools.Document(pdf_utils.repair_pdf(pdf_file), page_numbers=[page])
pdf = pydoxtools.Document(img)
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %%
#img

# %%
isinstance(img,PIL.Image.Image)

# %%
pdf.document_type

# %%
pdf.table_areas

# %%
[pretty_print(t) for t in pdf.tables_df]

# %%
page=0
boxes, box_levels = pdf.table_candidates, pdf.table_box_levels

# %%
vda.plot_box_layers(
    box_layers=[
        [pdf.line_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        [pdf.image_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False)],
        [pdf.graphic_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="yellow", filled=False)],
        
        #[box_levels[0][0].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
        #[box_levels[0][1].values if len(box_levels[18])>1 else [], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        #[pd.DataFrame([b._initial_area for b in boxes]).values, vda.LayerProps(alpha=1.0, color="red", filled=False, box_numbers=True)]
        [pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        #[figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250,
    #image=pdf.images[page],
    #image_box=pdf.pages_bbox[page],
),

# %%
