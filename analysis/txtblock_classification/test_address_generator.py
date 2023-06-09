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

# %%
import pydoxtools.random_data_generators
# %% [markdown]
# # Test business address generation

# %%
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
import collections

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
gen = pydoxtools.random_data_generators.BusinessAddressGenerator(type="address",rand_str_perc=0.5, osm_perc=0.5, fieldname_prob=0.1)

# %%
print(gen[random.random()])

# %%
#addr = [gen[random.random()] for i in tqdm(range(10000000))]
addr = [gen[random.random()] for i in tqdm(range(1000))]

for a in random.sample(addr,100): print(f"_____\n{a}\n")

# %%
