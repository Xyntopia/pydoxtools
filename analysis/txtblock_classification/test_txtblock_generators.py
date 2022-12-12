# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Train the Textblock classifier

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, classifier, nlp_utils, cluster_utils, training
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

box_cols = cluster_utils.box_cols

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__, torch.backends.cudnn.version()

# %%
bg = training.TextBlockGenerator(generators=(
    ("address",training.BusinessAddressGenerator()),
    ("unknown",training.RandomTextBlockGenerator()),
    ("unknown",training.RandomListGenerator()),
),weights=[10,8,2],
augment_prob=0.05, cache_size=100,renew_num=10)
bg.classmap,bg.classmap_inv, bg.num_generators, bg.class_gen

# %%
bgi=bg.__iter__()

# %% [markdown]
# check how fast the text generation is wwith different cache settings...

# %%
# %%timeit
addr = [next(bgi) for i in range(1000)]
#for a in addr: print(f"{a}\n")

# %% [markdown]
# progression:
#
# - 144ms/1k
# - 140ms/1k with 100/100
# - 79ms/1k with 100/50
# - 14.1ms/1k with 100/10

# %%
for p in [next(bgi) for i in range(10)]:
    print(p[0]+"\n\n")

# %%
