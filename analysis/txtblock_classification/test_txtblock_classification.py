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
# # Test the how well pre-trained transformer modles work on zero-shot classification tasks

# %%
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pathlib import Path

import pydoxtools.random_data_generators
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
import warnings
warnings.filterwarnings('ignore')
label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
df = pd.read_excel(label_file)
df['class']=df['label']
df = df.fillna(" ")

# %%
count = df["label"].value_counts()
count

# %%
classes = count.index.to_list()
classes

# %%
from transformers import AutoTokenizer
mdir=settings.MODEL_DIR/"txtblockclassifier"
#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
from transformers import AutoModelForSequenceClassification
txtclassmodel = AutoModelForSequenceClassification.from_pretrained(mdir)

# %%
from transformers import pipeline
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}
model = pipeline("text-classification",model=txtclassmodel, tokenizer=tokenizer)
#MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
#MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli

# %%
# evaluate prediction
dfl=df.txt.to_list()

# %%
res = [model(s, padding=True, truncation=True, return_all_scores=True) for s in tqdm(dfl)]

# %%
r = res[4]
r

# %%
df[['proba_address','pred_ukn']]=[(r[0][0]['score'],r[0][1]['score']) for r in res]

# %%
y_pred=df.apply(lambda x: "address" if x["proba_address"]>0.5 else "unknown", axis=1)

# %%
#y_pred[y_pred!="address"]="unknown"

# %%
#y_pred = model.predict(dfl)
#target_names = list(model.classmap_.values())
df["label"] = y_true = df["label"].replace(dict(
    contact="unknown",
    phone_number="unknown",
    multiaddress="unknown",
    directions="unknown",
    company="unknown",
    country="unknown",
    url="unknown",
    list="unknown"
))

# %%
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
#classifier.make_tensorboard_compatible(classification_report(y_true, y_pred, output_dict=True))

# %%
#classification_report(y_true, y_pred, output_dict=True)

# %%
df['proba_address'].hist(bins=20)

# %%
df[df.label=="address"].proba_address.hist(bins=20)

# %%
df[df.label!="address"].proba_address.hist(bins=20)

# %% [markdown]
# true positives?

# %%
tp = df[(df.label=="address") & (df.proba_address > 0.5)][["txt","proba_address","label"]]
#pretty_print(tp)

# %% [markdown]
# false positives?

# %%
fp = df[(df.label!="address") & (df.proba_address > 0.5)][["txt","proba_address","label"]]
len(fp)

# %%
pretty_print(fp)

# %% [markdown]
# measure model operations

# %%
# #!pip install thop

# %% [markdown]
# which addresses were not recognized?

# %%
model([
"""
XCAM Limited
2 Stone Circle Road
Round Spinney
Northampton
NN3 8RF, UK


Tel: 44 (0)1604 673700
Fax: 44 (0)1604 671584
Web: www.xcam.co.uk
Email: info@xcam.co.uk
- 10 -
""",
"""
XCAM Limited
2 Stone Circle Road
Round Spinney
Northampton
NN3 8RF, UK


Tel: 44 (0)1604 673700
Fax: 44 (0)1604 671584
Web: www.xcam.co.uk
Email: info@xcam.co.uk
""",
"""
Mailing Address: Texas Instruments, Post Office Box 655303, Dallas, Texas 75265
Copyright © 2014, Texas Instruments Incorporated
""",
"""
Mailing Address: Texas Instruments, Post Office Box 655303, Dallas, Texas 75265
2014, Texas Instruments Incorporated
""",
"""
319 Charleigh Ford, Jr. Drive | Columbus, Mississippi 39701 | 662.798.4075 | starkaerospace.com
"""
])

# %% [markdown]
# false negatives

# %%
fn = df[(df.label=="address") & (df.proba_address < 0.5)]
fn["proba_address_%"]=fn["proba_address"].round(2)
pretty_print(fn[["txt","proba_address_%","label"]])
#pretty_print(fn[["txt","proba_address","label"]].style.format(precision=2))

# %% [markdown]
# ## testing the generator..
#
# which textblocks are problematic in the generator?

# %%
addgen= pydoxtools.random_data_generators.BusinessAddressGenerator()
addr = [addgen(random.random()) for i in range(100)]
dfa=pd.DataFrame(model(addr))
dfa["txt"]=addr
print(dfa[dfa[0]!="address"].shape)
pretty_print(dfa[dfa[0]!="address"])

# %%
