# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
import pydoxtools
from pydoxtools import pdf_utils, classifier, nlp_utils
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
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/2-NeON2_V5_N1K_0LG_Datasheet_LGxxxN1K-V5_201905_EN.e5.pdf"

# %%
file

# %%

# %%

# %%
doc = pydoxtools.load_document(file)

# %%
doc.text_block_classes[doc.text_block_classes['add_prob'] > 0.5].txt.tolist()

# %%
doc.textboxes
