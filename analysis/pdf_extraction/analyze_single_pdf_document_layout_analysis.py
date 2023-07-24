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
from pdfminer.layout import LAParams
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
page = 0
page_num=[page]
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"

print(pdf_file)

# %%
#pdf = pydoxtools.Document(pdf_utils.repair_pdf(pdf_file), page_numbers=[page])
pdf = pydoxtools.Document(pdf_utils.repair_pdf(pdf_file), page_numbers=[page])
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %% [markdown]
# ## manually extract text

# %% [markdown]
# ## plot textboxes & characters

# %%
pdf.images

# %%
# #%prun pdf.elements.iloc[0]

# %%
# #%%prun
#from pdfminer.high_level import extract_text
#text = extract_text(pdf_file, page_numbers=[page])

# %%
#images[1]

# %%
#import pydoxtools.extract_textstructure as ts
#df = gu.boundarybox_query(pdf.elements,[600,400,700,500])
#df = ts.group_elements(df, ["boxnum"], "boxes_from_lines_w_bb")
#print("\n".join(df.text))

# %%
vda.plot_boxes(
    pdf.elements[box_cols].values,
#    groups = dfl["hm"].values,
    #bbox = [600,400,700,500], 
    #dpi=250
)#p.page_bbox)

# %%
pdf.table_box_levels
candidate_areas = pdf.table_candidates[0].bbox

# %%
pdf.table_candidates[0].df

# %%
#print(pdf.full_text)

# %%
vda.plot_box_layers(
    box_layers=[
        [pdf.line_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        [pdf.image_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False)],
        [pdf.graphic_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="yellow", filled=False)],
        [pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        [candidate_areas, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        #[figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250,
    image=pdf.images[page],
    image_box=pdf.pages_bbox[page],
),

# %%
pdf.line_elements[pdf.line_elements.boxnum==70]

# %%
#print(pdf.full_text)

# %%
