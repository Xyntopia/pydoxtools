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
# # Test augmentation and mixing of textblocks
#
# import logging
#
# import torch
# from IPython.display import display, HTML
# from tqdm import tqdm

# %%
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, nlp_utils, cluster_utils, training
from pydoxtools.settings import settings
import logging
from tqdm import tqdm
import pandas as pd
import torch


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


logger = logging.getLogger(__name__)

box_cols = cluster_utils.box_cols

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__, torch.backends.cudnn.version()

# %%
bg = pydoxtools.random_data_generators.TextBlockGenerator.std_generator()
bg.classmap, bg.classmap_inv, bg.num_generators, bg.class_gen, bg.gen_mapping, bg.weights

# %%
bgi = bg.__iter__()

# %% [markdown]
# check how fast the text generation is wwith different cache settings...

# %%
# %%timeit
addr = [next(bgi) for i in range(1000)]
# for a in addr: print(f"{a}\n")

# %% [markdown]
# progression:
#
# - 21.6 ms 100/10

# %%
bg.classmap

# %%
next(bgi)

# %%
for p in [next(bgi) for i in range(100)]:
    print(f"{p[1]} {bg.classmap_inv[p[1]]}:\n{p[0]}\n\n")

# %%

# %%
