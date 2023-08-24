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
import sys

import componardo.documentx

sys.path.append("..")
# %load_ext autoreload
# %autoreload 2

import logging
import pydoxtools
import

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# %%
import pydoxtools
import numpy as np
import yaml
from pydoxtools import pdf_utils, file_utils, nlp_utils, cluster_utils, document_base
from componardo.settings import settings
import componardo.documentx
from componardo import extract_product as ce
import torch
import componardo.spec_utils as su
import random
import pathlib
from operator import attrgetter

import pandas as pd
from tqdm import tqdm
#from IPython.display import display, Markdown, Latex
import os
from os.path import join

tqdm.pandas()

pdf_utils._set_log_levels()

nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %%
