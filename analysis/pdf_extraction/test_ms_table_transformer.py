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

# %% [markdown]
# ## try to extract table boundaries using our own method
#
# 1. first we will group lines into boxes to identify "whole" cells the grouping was already provided by pdfminer.six --> **box_groups**
# 2. afterwards we will group those boxes into columns & rows --> **valid_columns**
# 3. as the third step, cluster the columns into tables --> table_groups

# %%
training_data = pathlib.Path.home() / "comcharax/data"
page = 0
page_num=[page]
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
#page_num=None
#pdf_file= training_data / "woodfield/Woodfield WestEdge.pdf"

print(pdf_file)

# %%
pdf=pydoxtools.Document(pdf_file, page_numbers=[page])
img = pdf.images[page]
pdf = pydoxtools.Document(img)

# %%
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

inputs = image_processor(images=img, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([img.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[
    0
]

results

# %%
size=pdf.pages_bbox[page][2:]
rendersize = pdf.images[page].size
ratio=size/np.array(pdf.images[page].size)
size, rendersize, ratio

# %%
areas_raw=results['boxes'].detach().numpy()
areas = pd.DataFrame(areas_raw*ratio[0], columns=['x0','y0','x1','y1'])
#areas = (2*rendersize-areas)*ratio[0]
#np.array([size[1],0,0,0])-areas
areas[['y0','y1']]=size[1]-areas[['y0','y1']]
areas

# %%
vda.plot_box_layers(
    box_layers=[
        [pdf.line_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        [pdf.image_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False)],
        [pdf.graphic_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="yellow", filled=False)],
        [areas[box_cols].values, vda.LayerProps(alpha=0.5, color="green", filled=False)]
        #[box_levels[0][0].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
        #[box_levels[0][1].values if len(box_levels[18])>1 else [], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        #[pd.DataFrame([b._initial_area for b in boxes]).values, vda.LayerProps(alpha=1.0, color="red", filled=False, box_numbers=True)]
        #[pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        #[figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250,
    image=pdf.images[page],
    #image_box=pdf.pages_bbox[page],
),

# %%
margin=20
timg=img.crop(areas_raw[0]+(-margin,-margin,margin,margin))
timg

# %%
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

inputs = image_processor(images=timg, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([timg.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[
    0
]

results

# %%
model.config.id2label

# %%
cell_areas = results['boxes'].detach().numpy()
cell_areas

# %%
vda.plot_box_layers(
    box_layers=[
        #[pdf.line_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
        #[pdf.image_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="blue", filled=False)],
        #[pdf.graphic_elements[box_cols].values, vda.LayerProps(alpha=0.5, color="yellow", filled=False)],
        [cell_areas, vda.LayerProps(alpha=1.0, color="green", filled=False)]
        #[box_levels[0][0].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
        #[box_levels[0][1].values if len(box_levels[18])>1 else [], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        #[pd.DataFrame([b._initial_area for b in boxes]).values, vda.LayerProps(alpha=1.0, color="red", filled=False, box_numbers=True)]
        #[pdf.table_areas[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[tables[box_cols].values, vda.LayerProps(alpha=1.0, color="red", filled=False)],
        #[figures[box_cols].values, vda.LayerProps(alpha=1.0, color="green", filled=False)],
        #[text[box_cols].values, vda.LayerProps(alpha=1.0, color="blue", filled=False)],
        #[p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        #[t.df_ch[vda.box_cols].values, vda.LayerProps(alpha=1.0, color="yellow", filled=False)],
        #[t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.3, color="random", filled=True)]
    ],
    bbox=(0,0)+timg.size, dpi=250,
    image=timg,
    image_box=(0,0)+timg.size,
),

# %%
