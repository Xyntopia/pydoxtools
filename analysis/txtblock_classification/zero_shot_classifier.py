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
# # Test zero shot classification
# %%
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, nlp_utils, cluster_utils, file_utils
from pydoxtools.settings import settings
import logging
import torch
import pathlib
import numpy as np
import pydoxtools


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


logger = logging.getLogger(__name__)

box_cols = cluster_utils.box_cols

nlp_utils.device, torch.cuda.is_available(), torch.__version__, torch.backends.cudnn.version()

# %%
training_data = pathlib.Path.home() / "comcharax/data"
# get all pdf files in subdirectory
# files = file_utils.get_nested_paths(training_data / "pdfs/datasheet", "*.pdf")
file = "/home/tom/git/doxcavator/backend/lib/componardo/tests/data/Datasheet-Centaur-Charger-DE.6f.pdf"
files = file_utils.get_nested_paths(training_data / "sparepartsnow", "*.pdf")
len(files)

# %%
# pdf_file=random.choice(files)
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %%
pages =np.arange(10,15).tolist()
pages =[10,14,18,19] # we have an unreasonable number of elements here..  what is going on?
#pages = [18]
pdf = pydoxtools.Document(pdf_file, page_numbers=pages)


# %%
pdf.page_classifier(["product_description","tables_with_product_specifications"])

# %%
print(pdf.page_templates[19])

# %%
