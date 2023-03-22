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
import pydoxtools.random_data_generators
# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, classifier, nlp_utils, cluster_utils, training, file_utils
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings
import torch
import gzip
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
if False: #merge wet files from common crawl into a single string
    dn ="/home/tom/comcharax/data/raw_text"
    fs = file_utils.get_all_files_in_nested_subdirs(dn, '*.wet')
    merged_txt = ""
    for fn in fs:
        with open(fn) as f:
            txt = f.read()
            merged_txt+=re.sub("WARC\/1\.0(\n(WARC|Content)[^\n]*)+","",txt)

# %%
if False: #save as binary text file
    with open("/home/tom/comcharax/data/raw_text/all_text.txt","wb") as f:
        f.write(merged_txt.encode('utf-8'))

# %%
tg = pydoxtools.random_data_generators.RandomTextBlockGenerator(txt_source=settings.TRAINING_DATA_DIR / "all_text.txt")

# %%
#txt = tg([random.random() for i in range(10)])
txt = tg(list(range(200)))
for t in txt:
    print(f"{t}\n")

# %%
