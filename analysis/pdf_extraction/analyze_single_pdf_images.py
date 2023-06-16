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
# # Analyze the extraction of text from pdfs
#
# Check out how characters & textlines/textboxes are getting extracted from pdfs. For example
# their positions

# %%
import sys

sys.path.append("..")
# %load_ext autoreload
# %autoreload 2

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# %%
from pydoxtools import nlp_utils
from pydoxtools import pdf_utils
from pydoxtools import visual_document_analysis as vda
from pydoxtools import cluster_utils as gu
import numpy as np
import pandas as pd

import pydoxtools.cluster_utils
box_cols = pydoxtools.cluster_utils.box_cols
from pydoxtools.settings import settings
import torch
import pathlib
from IPython.display import display, HTML

from tqdm import tqdm

tqdm.pandas()

pdf_utils._set_log_levels()


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %%
x0,y0,x1,y1 = 1,2,3,4

# %% [markdown]
# ## first of all, get some correctly identified areas

# %% [markdown]
# For every pdf where text extraction doesn't work its definitly worth checking out
# how the pdf looks like by trying to repair it using something like the following:
#
#     gs -o repaired.pdf -sDEVICE=pdfwrite output.pdf

# %%
training_data = pathlib.Path.home() / "comcharax/data"
pages = [11,15]
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %%
#pdf = pydoxtools.Document(pdf_utils.repair_pdf(pdf_file), page_numbers=[page])
pdf = pydoxtools.Document(pdf_utils.repair_pdf(pdf_file), page_numbers=pages)
# pdfi.tables
# p.tables
# pdfi.table_metrics_X()

# %%
#pdfi.table_metrics
#pdfi.table_metrics_X

# %%
#[pretty_print(t) for t in pdfi.tables_df]

# %%
#pdfi.tables_df[0]

# %% [markdown]
# ## manually extract text

# %% [markdown]
# ## plot textboxes & characters

# %%
pdf.images

# %%
pdf.images[15]

# %%
