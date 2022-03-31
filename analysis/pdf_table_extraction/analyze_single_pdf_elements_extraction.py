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
# # Analyze the extraction of text from pdfs
#
# Check out how characters & textlines/textboxes are getting extracted from pdfs. For example
# their positions

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

# %% [markdown]
# For every pdf where text extraction doesn't work its definitly worth checking out
# how the pdf looks like by trying to repair it using something like the following:
#
#     gs -o repaired.pdf -sDEVICE=pdfwrite output.pdf

# %%
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/2239470.28.pdf"
print(pdf_file)

# %%
pdfi = pdf_utils.PDFDocument(pdf_file)
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %%
#pdfi.table_metrics
#pdfi.table_metrics_X

# %%
#[pretty_print(t) for t in pdfi.tables_df]

# %%
#pdfi.tables_df[0]

# %% [markdown]
# ## manually extract text

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## plot textboxes & characters

# %%
t = pdfi.table_objs[0]
t.df_words

# %%
t.df

# %% tags=[]
# vda.plot_boxes(
#    dfl[vda.box_cols].values,
#    groups = dfl["hm"].values,
#    bbox = None, dpi=250)#p.page_bbox)

print(pdf_file)
p = pdfi.pages[0]
area = pdfi.table_objs[0].bbox + [-10, -10, 10, 10]
#p = pdfi.pages[page]
boxes, box_levels = p.detect_table_area_candidates()
images = vda.cached_pdf2image(pdfi.fobj)

ERROR = np.array([[199.79,  522.362, 291.686, 579.698]])

vda.plot_box_layers(
    box_layers=[
        [p.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red", filled=False)],
        [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        [t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=area, dpi=250,
    image=images[0],
    image_box=p.page_bbox,
),

# %%

# %%
