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
page = 15
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
import layoutparser as lp
model = lp.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )
layout = model.detect(pdf.images[page])

# %%
import layoutparser as lp
model = lp.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', # In model catalog
            label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )
layout = model.detect(pdf.images[page])

# %%
size=pdf.pages_bbox[page][2:]
rendersize = pdf.images[page].size

# %%
ratio=size/np.array(pdf.images[page].size)

# %%
layout

# %%
df=layout.to_dataframe().rename(columns={'x_1':'x0', 'y_1':'y0', 'x_2':'x1','y_2': 'y1'})
df[box_cols]=((rendersize+rendersize)-df[box_cols])*ratio[0]
tables=df[df.type=="Table"]
figures=df[df.type=="Figure"]
text = df[df.type=="Text"]

# %%
raise

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

# %% jupyter={"outputs_hidden": true}
vda.plot_boxes(
    pdf.elements[box_cols].values,
#    groups = dfl["hm"].values,
    #bbox = [600,400,700,500], 
    #dpi=250
)#p.page_bbox)

# %%
elements=pdf.line_elements[pdf.line_elements.rawtext.str.strip()!=""]
#elements

# %%
vda.plot_box_layers(
    box_layers=[
        #[elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        [tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        [figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        [text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250,
    image=pdf.images[page],
    image_box=pdf.pages_bbox[page],
),

# %%
