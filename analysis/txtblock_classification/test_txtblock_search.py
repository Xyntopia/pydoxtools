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
# # Search for addresses in all kinds of documents
#
# - generate addresses & non-address textblocks
# - measure quality of algorithm
# - identify mis-identified addresses

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pathlib import Path
from pydoxtools import pdf_utils, classifier, nlp_utils, cluster_utils, training
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

box_cols = cluster_utils.box_cols

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
TRAINING_DATA_DIR = Path("../../training_data").resolve() #settings.TRAINING_DATA_DIR
label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"

# %%
df_labeled = pd.read_excel(label_file)

# %%
settings.TRAINING_DATA_DIR

# %%
df_labeled.label.unique()

# %%
model = classifier.load_classifier("text_block")

# %% tags=[]
import warnings
warnings.filterwarnings('ignore')
label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
df = pd.read_excel(label_file)
df['class']=df['label']
df = df.fillna(" ")

# %%
count = df["class"].value_counts()
count

# %%
classes = count.index.to_list()
classes

# %%
# evaluate prediction
dfl=df.txt.to_list()
y_pred = model.predict(dfl)
y_true = df["class"].replace(dict(
    contact="unknown",
    directions="unknown",
    company="unknown",
    country="unknown",
    url="unknown",
    list="unknown"
))
target_names = list(model.classmap_.values())

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=target_names))

# %%
set(y_pred), y_true.unique()

# %%
model.classmap_,model.predict_proba(dfl).detach().numpy()

# %%
df['proba'] = model.predict_proba(dfl).detach().numpy()[:,0]

# %%
df['proba'].hist()

# %%
df[df.label=="address"].proba.hist()

# %% [markdown]
# false positives?

# %% tags=[]
pretty_print(df[(df.label!="address") & (df.proba > 0.5)][["txt","proba","label"]])

# %% [markdown]
# which addresses were not recognized?

# %%
model.predict_proba(["""
MEAN WELL USA, INC.
44030 Fremont Blvd., Fremont,
CA 94538,
Tel: +1-510-683-8886
Web: www.meanwellusa.com""",
"""
Available Exclusively From 
Less EMF Inc
www.lessemf.com
+1-518-432-1550""",
"""
XCAM Limited, </s>2 Stone Circle Road, Round Spinney, Northampton, NN3 8RF
""",
"""
Ⓒ 2018 LG Chem ESS Battery Division
LG Guanghwamun Building, 58, Saemunan-ro, Jongro-gu, Seoul, 03184, Korea
http://www.lgesspartner.com http://www.lgchem.com
"""
"""
XCAM Limited
2 Stone Circle Road
Northampton
NN3 8RF
Tel: 44 (0)1604 673700
Fax: 44 (0)1604 671584
www.xcam.co.uk
Email: info@xcam.co.uk
"""
])

# %% tags=[]
pretty_print(df[(df.label=="address") & (df.proba < 0.5)][["txt","proba","label"]])

# %% [markdown]
# ## testing the generator..
#
# which addresses were problematic in the generator?

# %%
bg = training.TextBlockGenerator(generators=dict(
    address=training.BusinessAddressGenerator(fake_langs=['en_US', 'de_DE', 'en_GB']),
    unknown=training.RandomTextBlockGenerator()
),augment_prob=0.05, cache_size=100,renew_num=10)
bg.classmap, bg.num_generators

# %%
bgi=bg.__iter__()
addr = [next(bgi) for i in range(1000)]

# %%
df = pd.DataFrame(addr, columns=["txt","class"])

# %%
df["class"]=df["class"]
df["class"]=df["class"].apply(lambda x: x.numpy())

# %%
# evaluate prediction
dfl=df.txt.to_list()
y_pred = model.predict(dfl)
y_true = df["class"].replace(dict(
    contact="unknown",
    directions="unknown",
    company="unknown",
    country="unknown"
))
target_names = list(model.classmap_.values())

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=target_names))

# %%
