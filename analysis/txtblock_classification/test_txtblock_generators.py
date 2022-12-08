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
gen = training.BusinessAddressGenerator(fake_langs=['en_US', 'de_DE', 'en_GB'])
tg = training.RandomTextBlockGenerator()
bg = training.TextBlockGenerator(generators=dict(
    address=gen,
    text=tg
),augment_prob=0.0)

# %%
bg.classmap, bg.num_generators

# %%
bgi=bg.__iter__()

# %%
# %%timeit
addr = [next(bgi) for i in range(64)]
#for a in addr: print(f"{a}\n")

# %%
#3.16s for 10k samples:
tha would be ~3.16s for a 10k batch thats pretty good...