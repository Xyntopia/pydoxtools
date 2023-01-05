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
import pandas as pd
#import dask
import numpy as np
#import dask.bag as db
import dask.dataframe as dd
#from dask.delayed import delayed
#from dask.distributed import Client
from tqdm import tqdm

# %%
#pip install fiona (in order to open the respecive file!)
#convert addresses from osmdata.xyz
if False:
    fn = "/home/tom/comcharax/data/trainingdata/osm/address_EPSG4326.gpkg"
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
if False: #convert our existing extracted addresses file to csv
    df = pd.read_parquet(
        "/home/tom/comcharax/data/osm_address_data.parquet", 
        engine="fastparquet"
    )#.sample(100000).copy()

    df.to_csv("/home/tom/comcharax/data/osm_address_data.csv", chunksize=1000)

# %%
ddf = dd.read_csv("/home/tom/comcharax/data/osm_address_data.csv", blocksize="64MB", dtype=str)

# %%
ddf.columns

# %% [markdown]
# columns not worth using:   addr, 

# %%
from dask.diagnostics import ProgressBar
with ProgressBar():
    df = ddf["addr_place"].unique().compute()
df

# %%
from dask.diagnostics import ProgressBar
with ProgressBar():
    cities = ddf["addr_city"].unique().compute()
    countries = ddf["addr_country"].unique().compute()
    streets=ddf["addr_street"].unique().compute()
    names=ddf["name"].unique().compute()

# %%
cities.to_csv("/home/tom/comcharax/data/cities.txt", index=False, header=False)
streets.to_csv("/home/tom/comcharax/data/streets.txt", index=False, header=False)
names.to_csv("/home/tom/comcharax/data/names.txt", index=False, header=False)
states.to_csv("/home/tom/comcharax/data/states.txt", index=False, header=False)

# %%
# most common words in "streets":
street_words = streets.str.split().explode().value_counts()
pd.Series(street_words.index).to_csv("/home/tom/comcharax/data/street_words.txt", index=False, header=False)

# %%
# most common words in "names":
name_words = names.str.split().explode().value_counts()
pd.Series(name_words.index).to_csv("/home/tom/comcharax/data/name_words.txt", index=False, header=False)

# %% [markdown]
# extract all countries

# %% [markdown]
# extract all street names with respective countries

# %%
if False:
    street_groups = df.groupby("addr_street", dropna=False)
    street_names = street_groups.apply(lambda x: 
        pd.Series({
            "count": len(x), 
            "countries": list(x.addr_country.dropna().unique())
        })
    )

# %% [markdown]
# explore the othe values...

# %%
df.apply(lambda x: x.dropna().unique())
