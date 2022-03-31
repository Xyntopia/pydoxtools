# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyze the extraction of tables from pdfs

# %% tags=[]
import sys

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

box_cols = pdf_utils.box_cols
from pydoxtools.settings import settings
import torch
from IPython.display import display, HTML

from tqdm import tqdm

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %%
x0,y0,x1,y1 = 1,2,3,4

# %% [markdown]
# ## first of all, get some correctly identified areas

# %%
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/PRF-PR-21_HRVI-2nd-gen.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/NEXXIS_TECH_SHEET_VT1000HDPTZATEX.34.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/T2000BrochureAUS_WEB.fa.pdf"
#pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Optocouplers.d3.pdf"
#pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/14-DD05A.08(II)_EN_TSM_DD05A_08_II_plus_datasheet_B_2017_web.22.pdf"
print(pdf_file)

# %%
params = pdf_utils.TableExtractionParameters.reduced_params()
#params.text_extraction_margin=20.0
pdfi = pdf_utils.PDFDocument(pdf_file, table_extraction_params=params)
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %% [markdown]
# ## calculate table text for the page

# %% tags=[]
page, table_candidate = 1,1
#pretty_print(pdfi.pages[page].tables[table])
#pdfi.pages[page].tables[table]
t = pdfi.pages[page].table_candidates[table_candidate]
#t = pdfi.pages[page].tables[table_candidate]

# %% tags=[]
pretty_print(t.df)
#t.df
#pdfi.pages[page]
t.is_valid

# %%
#pdfi.table_metrics
#pdfi.table_metrics_X

# %%
#[pretty_print(t) for t in pdfi.tables_df]

# %%
#pdfi.tables_df[0]

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## plot table candidate

# %%
margin=20
borders =b= 2
#ca = t.detect_cells()
ca = t.detect_cells()
layers = [
    [t.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.05, color="blue", filled=True)],
    #[t.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.2, color="orange")],
    [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.2, color="orange")],
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

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## plot table candidate areas

# %% tags=[]
# vda.plot_boxes(
#    dfl[vda.box_cols].values,
#    groups = dfl["hm"].values,
#    bbox = None, dpi=250)#p.page_bbox)

print(pdf_file)
#p = pdfi.pages[2]
p = pdfi.pages[page]
boxes, box_levels = p.detect_table_area_candidates()

ERROR = np.array([[199.79,  522.362, 291.686, 579.698]])

vda.plot_box_layers(
    box_layers=[
        [p.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red", filled=False)],
        [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        [box_levels[0].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
        [box_levels[1].values if len(box_levels) > 1 else [], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        [boxes.values, vda.LayerProps(alpha=0.1, color="blue", filled=False)],
        [p.table_candidate_boxes_df.values, vda.LayerProps(
            alpha=1.0, color="red", filled=False, box_numbers=True)]
        #[ERROR,vda.LayerProps(color="red")]
    ],
    bbox=p.page_bbox, dpi=250
),

# %%

# %%
