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

# %%
# %load_ext autoreload
# %autoreload 2
# #!pip install dask[complete]

# %%
import geopandas as gp
import pandas as pd
import fiona
import dask
import numpy as np
import dask.bag as db
import dask.dataframe as dd
from dask.delayed import delayed
from dask.distributed import Client
from tqdm import tqdm

# %%
#convert addresses from osmdata.xyz
fn = "/home/tom/comcharax/data/trainingdata/osm/address_EPSG4326.gpkg"
if False:
    size = 5000000
    last_save = 0
    with fiona.open(fn) as src:
        count = len(src)
        print(f'count: {count}')
        df = []
        for i,item in tqdm(enumerate(src)):
            df.append(item['properties'])
            if size<=(i-last_save):
                print(f"saving {i}")
                df = pd.DataFrame(df)
                df = df.drop(columns=['osm_id','osm_timestamp','other_tags'])
                df = df.to_parquet(f"/home/tom/comcharax/data/trainingdata/osm_addresses/addr_{i}.parquet")
                df = []
                last_save=i

    df = pd.DataFrame(df)
    df = df.drop(columns=['osm_id','osm_timestamp','other_tags'])
    df = df.to_parquet(f"/home/tom/comcharax/data/trainingdata/osm_addresses/addr_{i}.parquet")

# %%
df = pd.read_parquet("/home/tom/comcharax/data/trainingdata/osm_addresses", engine="fastparquet")

# %%
#df.describe()
#df = df.astype(pd.SparseDtype("string", np.nan))
#df.drop(columns=[""])
df.to_parquet("/home/tom/comcharax/data/trainingdata/address_data.parquet")

# %%
