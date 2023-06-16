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
# # Analyze the extraction of text from pdfs
#
# Check out how characters & textlines/textboxes are getting extracted from pdfs. For example
# their positions

# %%
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

import pydoxtools.cluster_utils
box_cols = pydoxtools.cluster_utils.box_cols
from pydoxtools.settings import settings
import torch
import pathlib
from IPython.display import display, HTML

from tqdm import tqdm

tqdm.pandas()

pdf_utils._set_log_levels()


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
training_data = pathlib.Path.home() / "comcharax/data"
page = 17
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %%
pdf = pydoxtools.Document(pdf_file, page_numbers=[page])
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

# %% [markdown]
# ## plot textboxes & characters

# %%
# %prun pdf.elements.iloc[0]

# %%
pdf.elements.boxnum.unique()

# %%
pdf.elements.shape

# %%
#print(pdf.full_text)

# %%
pdf.elements[box_cols].values

# %%
import pdf2image
images = pdf2image.convert_from_path(pdf.fobj, dpi=240,first_page=16,last_page=17, use_cropbox=True)
images[1]

# %%
import pydoxtools.extract_textstructure as ts
df = gu.boundarybox_query(pdf.elements,[600,400,700,500])
df = ts.group_elements(df, ["boxnum"], "boxes_from_lines_w_bb")
print("\n".join(df.text))

# %%
boundarybox_query

# %%
vda.plot_boxes(
    df.values,
#    groups = dfl["hm"].values,
    bbox = [600,400,700,500], 
    dpi=250
)#p.page_bbox)

# %%
#print(pdf_file)
p = pdfi.pages[0]
area = pdfi.table_objs[0].bbox + [-10, -10, 10, 10]
#p = pdfi.pages[page]
boxes, box_levels = p.detect_table_area_candidates()

# %%
images[page]

# %%
ERROR = np.array([[199.79,  522.362, 291.686, 579.698]])

vda.plot_box_layers(
    box_layers=[
        [p.df_ge[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=area, dpi=250,
    image=images[page],
    image_box=p.page_bbox,
),
