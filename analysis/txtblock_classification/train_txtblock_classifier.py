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
from pathlib import Path
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

# %% [markdown] tags=[]
# test the model once

# %%
if False:
    _,_,m = training.prepare_textblock_training()
    res =m.predict_proba(["""ex king ltd
    Springfield Gardens
    Queens
    N. Y 11413
    www.something.com
    """,
     """ex king ltd
         Springfield Gardens
         Queens
         N. Y 11413
         www.something.com
         """
     ])
    print(res)

# %% [markdown]
# TODO: its probabybl a ood idea to use some hyperparemeter optimization in order to find out what is the best method here...
#
# we would probably need some manually labeled addresses from our dataset for this...

# %%
#classifier.load_labeled_text_blocks.clear()

# %% tags=[]
import warnings
warnings.filterwarnings('ignore')
label_file = Path("../../training_data/labeled_txt_boxes.xlsx")
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
#from sklearn.utils import compute_class_weight
weights = sklearn.utils.class_weight.compute_class_weight('balanced',classes = classes, y=df["class"])
weights

# %%
pretty_print(df[df["class"] == "address"].sample(10))

# %% [markdown]
# start training

# %%
if False:
    # evaluate prediction
    train_loader, test_loader, model = training.prepare_textblock_training(4)
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

# %% tags=[]
# %env TOKENIZERS_PARALLELISM=true
# url of nextcloud instance to point to
hostname = 'https://sync.rosemesh.net'
# the token is the last part of a sharing link:
# https://sync.rosemesh.net/index.php/s/KwkyKj8LgFZy8mo   the  "KwkyKj8LgFZy8mo"  webdav
# takes this as a token with an empty password in order to share the folder
token = "KwkyKj8LgFZy8mo"
syncpath = str(settings.MODEL_DIR)
upload = False

# %%
# test webdav connection
# wu.push_dir_diff(hostname, token, syncpath)


# %% tags=[]
class WebdavSyncCallback(pytorch_lightning.Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wu.push_dir_diff(hostname, token, syncpath)

additional_callbacks = []
if upload:
    additional_callbacks = [WebdavSyncCallback()]

trainer, model = training.train_text_block_classifier(
    num_workers=8,
    max_epochs=-1, gpus=1, callbacks=additional_callbacks,
    steps_per_epoch=50
)

# %%
