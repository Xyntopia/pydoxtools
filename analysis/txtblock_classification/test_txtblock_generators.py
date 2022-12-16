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
# # Test augmentation and mixing of textblocks

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
),weights=[100,80,20],
random_char_prob=0.1, random_word_prob=0.1,
cache_size=100,renew_num=10, mixed_blocks_generation_prob=0.1, mixed_blocks_label="unknown")
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
# - 28.2 ms 100/10

# %%
for p in [next(bgi) for i in range(100)]:
    print(f"{bg.classmap[p[1].item()]}:\n{p[0]}+\n\n")

# %%
