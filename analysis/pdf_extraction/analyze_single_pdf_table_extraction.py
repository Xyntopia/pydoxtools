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
box_cols = pdf_utils.box_cols
from pydoxtools.settings import settings
import torch
from IPython.display import display
import pdfminer
import pdfminer.layout
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
memory = settings.get_memory_cache()

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
# ## try to extract table boundaries using my own method
#
# 1. first we will group lines into boxes to identify "whole" cells the grouping was already provided by pdfminer.six --> **box_groups**
# 2. afterwards we will group those boxes into columns & rows --> **valid_columns**
# 3. as the third step, cluster the columns into tables --> table_groups

# %%
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/BUSI-XCAM-SY-00010.36.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Optocouplers.d3.pdf"
print(pdf_file)

# %%
pdfi = pdf_utils.PDFDocumentOld(pdf_file)
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %%
[pretty_print(t) for t in pdfi.tables_df]

# %%
#pdfi.tables_df

# %%
m = pdfi.table_metrics_X

# %%
# vda.plot_boxes(
#    df_le[vda.box_cols].values,
#    groups = df_le["hm"].values,
#    bbox = None, dpi=250)#p.page_bbox)
#p = pdfi.pages[0]
for p in pdfi.pages:
    vda.plot_box_layers(
        box_layers=[
            [p.df_ge_f[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red")],
            [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="black")],
            [p.detect_table_area_candidates()[0].values, vda.LayerProps(alpha=0.2, color="blue", filled=False)],
            [p.table_candidate_boxes_df.values, vda.LayerProps(
                alpha=0.5, color="red", filled=False, box_numbers=True)],
            [p.table_areas.values, vda.LayerProps(alpha=0.2, color="yellow")]
        ],
        bbox=p.page_bbox, dpi=250
    ),print(pdfi.filename)

# %%

# %%
