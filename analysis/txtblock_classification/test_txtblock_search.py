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
count = df["label"].value_counts()
count

# %%
classes = count.index.to_list()
classes

# %%
# evaluate prediction
dfl=df.txt.to_list()
y_pred = model.predict(dfl)
df["label"] = y_true = df["label"].replace(dict(
    contact="address",
    multiaddress="unknown",
    directions="unknown",
    company="unknown",
    country="unknown",
    url="unknown",
    list="unknown"
))
target_names = list(model.classmap_.values())

# %%
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
#classifier.make_tensorboard_compatible(classification_report(y_true, y_pred, output_dict=True))

# %%
model.classmap_,model.predict_proba(dfl).detach().numpy()

# %%
df['proba_address'] = model.predict_proba(dfl).detach().numpy()[:,model.classmapinv_['address']]

# %%
df['proba_address'].hist(bins=20)

# %%
df[df.label=="address"].proba_address.hist(bins=20)

# %% [markdown]
# true positives?

# %% tags=[]
tp = df[(df.label=="address") & (df.proba_address > 0.5)][["txt","proba_address","label"]]
#pretty_print(tp)

# %% [markdown]
# false positives?

# %% tags=[]
fp = df[(df.label!="address") & (df.proba_address > 0.5)][["txt","proba_address","label"]]
len(fp)

# %% tags=[]
pretty_print(fp)

# %% [markdown]
# which addresses were not recognized?

# %%
model.predict_proba([
"""
north
2234 Circle Road
Bonn
NN3 8RF
""",
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
,
"""
XCAM Limited
2 Stone Circle Road
Northampton
NN3 8RF
Tel: 44 (0)1604 673700
Fax: 44 (0)1604 671584
www.xcam.co.uk
Email: info@xcam.co.uk
""",
"""
Fluke GmbH
Engematten 14
79286 Glottertal
Telefon: (069) 2 22 22 02 00
Telefax: (069) 2 22 22 02 01
E-Mail: info@de.ﬂuke.nl
Web: www.ﬂ uke.de
""",
"""
Fluke GmbH
Engematten 14
79286 engen
069 2 22 22 02 00
069 2 22 22 02 01
info@de.ﬂuke.nl
www.ﬂ uke.de
""",
"""
Astro- und Feinwerktechnik Adlershof GmbH
Albert Einstein Str. 12
12489 Berlin
""",
"""
Adlershof GmbH
Einstein Str. 12
12489 Berlin
""",
"""
Rosen Str. 12
12489 Berlin
""",
"""
Neue Str. 12
Berlin 12489
""",
"""
Astro-und Feinwerktechnik Adlershof GmbH
Albert-Einstein-Str. 12
12489 Berlin
Germany
""","""
Texas Instruments, Post Office Box 655303, Dallas, Texas 75265
Copyright © 2010, Texas Instruments Incorporated
""",
"""
Texas Instruments, Post Office Box 655303, Dallas, Texas 75265
"""
]), model.classmap_

# %% [markdown]
# false negatives

# %% tags=[]
fn = df[(df.label=="address") & (df.proba_address < 0.5)]
pretty_print(fn[["txt","proba_address","label"]])

# %% [markdown]
# ## testing the generator..
#
# which textblocks are problematic in the generator?

# %%
addgen=training.BusinessAddressGenerator()
addr = [addgen(random.random()) for i in range(10000)]
dfa=pd.DataFrame(model.predict(addr))
dfa["txt"]=addr
dfa[dfa[0]!="address"]

# %%
