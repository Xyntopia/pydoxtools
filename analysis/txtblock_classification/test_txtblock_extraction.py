# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, classifier, nlp_utils
from pydoxtools import geometry_utils as gu
from pydoxtools.visual_document_analysis import plot_boxes
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings
import torch
from IPython.display import display
import re
import random
import pytorch_lightning
import logging

from IPython.display import display, HTML
import pandas as pd
from tqdm import tqdm
from faker import Faker
import sklearn
import numpy as np
import os
from os.path import join


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


logger = logging.getLogger(__name__)

box_cols = pdf_utils.box_cols

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__, torch.backends.cudnn.version()

# %% [markdown]
# ## load pdf files

# %% [markdown]
# we can find addresses here:
#
# https://archive.org/details/libpostal-parser-training-data-20170304
#
# from this project: https://github.com/openvenues/libpostal
#
# now we can simply mix addresses from taht repository with random text boxes and
# run a classifier on them! yay!

# %% [markdown]
# # translate text boxes into vectors...

# %% [markdown]
# TODO: its probabybl a ood idea to use some hyperparemeter optimization in order to find out what is the best method here...
#
# we would probably need some manually labeled addresses from our dataset for this...

# %% [markdown]
# training...

# %%
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Datenblatt_PSL-Family.37.pdf"
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/remo-m_fixed-wing.2f.pdf"
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/MKPU-XCAM-MS-00042.4d.pdf"

# %%
file

# %%
pdf = pdf_utils.PDFDocument(file)

# %%
page = pdf.pages[1]
df = page.df_le.copy()
df.font_infos = df.font_infos.apply(list)
df = df.explode("font_infos")
df.font_infos.apply(list)
df[["fontname","lineheight","color"]]=df.font_infos.apply(pd.Series)
df.lineheight=df.lineheight.round(2)
df.lineheight.unique()

# %%
df.columns

# %%
line_feature_cols = ["x0","y0","x1","y1","fontname","lineheight","color"]

# %%
#cluster the lines using all their features combined...

# %%
features = pd.get_dummies(df[line_feature_cols])

# %%
feature_num = df.nunique()
feature_num

# %%
features.columns

# %%
data = features[:].values

# %%
# create weights for our distance function...

# TODO: in addition to weights we can define a "threshold"

weights = dict(
    coords=1.0,
    lineheight=2.0,
    fontname=0.5,
    color=0.5
)

weightlen = data.shape[1] #number of weights needed
weight_vec = [weights['coords']] + [weights['lineheight']] \
          + [weights['fontname']]*feature_num['fontname'] +\
            [weights['color']]*feature_num['color']
np.array(weight_vec)

# %%
dm = gu.calc_pairwise_matrix(gu.pairwise_box_area_distance_func, data[:, :4],
                             diag=0.0)
dm


# %%
def distance_func(data):
    dm = gu.calc_pairwise_matrix(gu.pairwise_box_area_distance_func, data, diag=0.0)
    return dm

groups, distance = gu.distance_cluster(
    data[:,:4], tol=15.0, pairwise_distance_func=gu.pairwise_box_edge_similarity_func)

# %%
groups

# %%
boxes = bx = features[box_cols].values
#groups = df.boxnum.to_list()
ax = plot_boxes(boxes, page.page_bbox, groups=groups)

# %%
txtblocks = pd.DataFrame(pdf_utils.get_pdf_text(file, boxes=True), columns=["txt"])
pretty_print(txtblocks)

# %%
