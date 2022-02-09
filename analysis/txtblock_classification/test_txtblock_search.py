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

# %% [markdown] tags=[]
# # Search for addresses in all kinds of documents

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, classifier, nlp_utils
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings
import torch
from IPython.display import display
import re
import random
import pytorch_lightning
import datetime
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

# %%
# df = classifier.get_address_collection()
# classifier.load_labeled_text_blocks.clear()
# df = classifier.load_labeled_text_blocks(cached=True)
#df[df["class"] == "address"].shape
#pretty_print(df[df["class"] == "address"].sample(10))
#df['class'].value_counts()

# %% [markdown]
# and test the model

# %%
label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.odf"

# %%
df_labeled = pd.read_excel(label_file, engine="odf")

# %%
df_labeled

# %%
import warnings
warnings.filterwarnings('ignore')
df = classifier.get_pdf_text_boxes()

# %%
model = classifier.load_classifier("text_block")

# %%
df = df.drop_duplicates("txt")
df.txt = df.txt.str.strip()
df["pred"] = df.txt.progress_apply(lambda x: model.predict_proba([x])[0])
df[["add_prob", "ukn_prob"]] = df.pred.progress_apply(pd.Series)

# %%
min_prob = 0.1
(df.add_prob>min_prob).sum(),df[df.add_prob>min_prob].add_prob.hist(bins=50)
