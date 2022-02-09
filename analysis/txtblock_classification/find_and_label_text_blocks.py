# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
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
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from IPython.display import display, HTML
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
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
# # classify textboxes

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
import warnings
warnings.filterwarnings('ignore')
dfn = classifier.get_pdf_text_boxes()

# %%
model = classifier.load_classifier("text_block")

# %%
model.cv2.bias.max()

# %%
dfn = dfn.drop_duplicates("txt")
dfn.txt = dfn.txt.str.strip()
dfn["pred"] = dfn.txt.progress_apply(lambda x: model.predict_proba([x])[0])
dfn[["add_prob", "ukn_prob"]] = dfn.pred.progress_apply(pd.Series)

# %%
label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"

# %%
min_prob = 0.05
(dfn.add_prob>min_prob).sum(),dfn[dfn.add_prob>min_prob].add_prob.hist(bins=100)

# %% [markdown] tags=[]
# ## compare classification with already labeled data

# %%
# get labeled dataset
df_labeled = pd.DataFrame()
df_labeled = pd.read_excel(label_file)
df_labeled

# %%
# generate some statistics on quality of detection...
# check which txt blocks are in the data AND were already labeled
dfboth = dfn.merge(df_labeled, on=['txt'], indicator=True, how="outer", suffixes=(None, "_labeled")).query('_merge=="both"')

# %%
dfboth

# %%
# get predicted vs. manually labeled examples 
y_true = dfboth.label=="address"
y_pred = dfboth.add_prob

# find the optimum threshold for maximum harmonic-average (macro) f1 score
res = minimize_scalar(
    lambda x: 1.0/f1_score(y_true, y_pred>x, average='macro'),
    bounds=(0, 1), method='bounded'
)
th_optim = res.x
th_optim

# %%
# number of detected addresses in files:
# True positives:
th = th_optim
Tp = dfboth[(dfboth.add_prob>th) & (dfboth.label=="address")]
Fp = dfboth[(dfboth.add_prob>th) & (dfboth.label=="unknown")]
Tn = dfboth[(dfboth.add_prob<th) & (dfboth.label=="unknown")]
Fn = dfboth[(dfboth.add_prob<th) & (dfboth.label=="address")]
stats = {
    "True Positives": len(Tp),
    "False Positives": len(Fp),
    "True Negatives": len(Tn),
    "False Negatives": len(Fn)
}
stats

# %% [markdown]
# current state:
#
#     {'True Positives': 43,
#      'False Positives': 77,
#      'True Negatives': 1799,
#      'False Negatives': 77}
#      
#      {'True Positives': 67,
#      'False Positives': 50,
#      'True Negatives': 2008,
#      'False Negatives': 65}
#      
#      {'True Positives': 66,
#      'False Positives': 48,
#      'True Negatives': 2011,
#      'False Negatives': 60}
#      
#     {'True Positives': 69,
#      'False Positives': 67,
#      'True Negatives': 2245,
#      'False Negatives': 59}

# %%
print(classification_report(y_true, y_pred>th_optim, target_names=["unknown","address"]))

# %%
display = PrecisionRecallDisplay.from_predictions(
   y_true, y_pred)
_ = display.ax_.set_title("txtblock 2-class Precision-Recall curve")

# %% [markdown]
# ## which ones were wrongly classified?

# %%
#Fp.txt.values #strings that were wrongly classified as address
#Fn.txt.values #addresses that were not recognized

# %% [markdown]
# ## generate new labeling data

# %%
# remove all "already-labeled" components
dfm = dfn.merge(df_labeled, on=['txt'], indicator=True, how="outer", suffixes=(None, "_labeled")).query('_merge=="left_only"')

# %%
dfm

# %%
min_prob = 0.9
(dfm.add_prob>min_prob).sum()

# %%
to_label = dfm[dfm.add_prob > min_prob].copy()
to_label = to_label[["txt", "filename"]]

# %%
df = to_label.merge(df_labeled, on=['txt'], indicator=True, how="outer", suffixes=(None, "_labeled"))

# %%
df["label"]=df["label"].fillna("TODO")
df["filename"]=df["filename"].fillna(df["filename_labeled"])
df

# %%
df._merge.value_counts()

# %%
label_backup = label_file.parent / (label_file.stem + datetime.datetime.today().strftime('%Y%m%d%H%M') + label_file.suffix)
label_backup

# %%
raise Exception("wait before saving!!!")

# %%
save_cols = ["txt","label","filename"]
# we ahve to use excel here, because pandas odf engine doesn't read newlines...
df[save_cols].drop_duplicates("txt").to_excel(label_file, header=True, index=False)
df[save_cols].drop_duplicates("txt").to_excel(label_backup, header=True, index=False)

# %%

# %%

# %%
#pretty_print(txtboxes[txtboxes.add_prob >= 0.99])
