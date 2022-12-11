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
# # Initialize training datasets
#
# - augmentation
# - cleaning
# - downloading new data
# - restricting to correct size

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
from pathlib import Path

# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, classifier, nlp_utils, list_utils, cluster_utils, training
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings
import torch
from IPython.display import display
import re
import random
import pytorch_lightning
import math
import logging
import concurrent.futures
import multiprocessing as mp

import matplotlib.pyplot as plt
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

# %% [markdown]
# weibul distributed chunks:

# %%
k, l = 1.4, 8
x = np.linspace(0,30,100)
plt.plot(x, k/l*(x/k)**(k-1)*np.exp(-(x/l)**k))


# %%
def task(rh):
    txt = classifier.html_clean(rh, 2000000)
    return txt


max_files = -1
parquet_dir = "html/page_solar"  # "pages.parquet"
TRAINING_DATA_DIR = Path("../../../componardolib/training_data").resolve() #settings.TRAINING_DATA_DIR

# %%
html_text = list()
filename = TRAINING_DATA_DIR / "html_text.parquet"

# %%
if False:
    # get a list of html text strings
    df = pd.read_parquet(str(settings.DATADIR / parquet_dir))
    df = df[df.html.str.len() > 0].reset_index(drop=True)
    html_list = df.html[:max_files]
    
    # extract text from a list of html strings for NLP tasks...
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for fn, txtbox_list in zip(html_list, tqdm(executor.map(
                # pdf_utils.get_pdf_text_safe_cached, files[:max_filenum]
                task,
                html_list.values
        ))):
            html_text.append(txtbox_list)

    pd.DataFrame(html_text, columns=["txt"]).to_parquet(filename)

# %%
# test if we can generate random augmented textblocks from it...
txt = classifier.generate_random_textblocks(filename)

# %%
if True:
    df = pd.read_csv(
        "/home/tom/comcharax/data/trainingdata/formatted_addresses_tagged.random.tsv",
        sep="\t",
        nrows=100,
        names=["country", "lang", "address"],
        #skiprows=lambda i: i > 0 and random.random() > 0.5
    )
    #df.loc[address_len.index[:5]]
    #df = df.loc[df.address.str.len()<275]
    df.to_parquet("/home/tom/comcharax/data/trainingdata/random_addresses.parquet")

# %%
# !pip install overpass

# %%
import overpass
#api = overpass.API(endpoint="http://lz4.overpass-api.de/api/",timeout=100, debug=True)
api = overpass.API(timeout=2000, debug=True)

# %% tags=[]
result = api.get("""
area[name="Deutschland"];
relation(area)["boundary"="administrative"][admin_level=4];
""", responseformat="csv(name)")
len(result),",".join(i[0] for i in result)

# %%
result = api.get("""
area[name="Deutschland"];
way(area)[highway][name];
for (t["name"])
(
  make x name=_.val;
  out; 
);
""", responseformat="csv(name)")

# %% [markdown]
# filter out addresses from ismand dataset

# %%
# #!pip install dask[complete]

# %%
import geopandas as gp
import pandas as pd
import fiona
import dask
import dask.bag as db
import dask.dataframe as dd
from dask.delayed import delayed

# %%
from dask.distributed import Client
client = Client()

# %%
client

# %%
fn = "/home/tom/comcharax/data/trainingdata/osm/address_EPSG4326.gpkg"


# %% [markdown]
# convert addresses from osmdata.xyz

# %%
@delayed
def fionapaginator(start,end):
    print(f"processing {start,end}")
    with fiona.open(fn) as src:
        df = []
        for item in src[start:end]:
            df.append(item['properties'])
    return pd.DataFrame(df)


# %%
size = 100000
with fiona.open(fn) as src:
    count = len(src)
    print(f'count: {count}')
    pagination = [(i,i+size) for i in range(0,len(src),size)]
    i=next(src)
    columns = {p:str for p in i['properties'].keys()}

# %%
df = dd.from_delayed([fionapaginator(start,end) for start,end in pagination])
df = df.drop(columns=['osm_id','osm_timestamp','other_tags'])
df.repartition(50).to_parquet("/home/tom/comcharax/data/trainingdata/osm_addresses.parquet")

# %% [markdown]
# load generated address dataset!

# %% tags=[]
df = training.get_address_collection()

# %%
#df

# %%
address_len = df.address.str.len().sort_values(ascending=False)

# %%
address_len.hist(bins=270)

# %% tags=[]
df = df[df.lang.isin(("de", "us", "en"))]
#pretty_print(df)

# %%
x = df.sample(1)
pretty_print(x)

# %%
file = "file:///home/tom/Dropbox/company/src/pydoxtools/training_data/promptcloud-amazon-product-details/data/marketing_sample_for_amazon_com_ecommerce_products_details_20190601_20190630_30k_data.csv"
file = "file:///home/tom/Dropbox/company/src/pydoxtools/training_data/promptcloud-amazon-product-details/original/marketing_sample_for_amazon_com-ecommerce_products_details__20190601_20190630__30k_data.csv/home/sdf/marketing_sample_for_amazon_com-ecommerce_products_details__20190601_20190630__30k_data.csv"
df = pd.read_csv(file)
