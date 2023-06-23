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
# # Analyze the extraction of table candidate areas from pdfs

# %%
import sys

import pydoxtools.extract_tables

sys.path.append("..")
# %load_ext autoreload
# %autoreload 2

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# %%
from pydoxtools import nlp_utils, Document
from pydoxtools import pdf_utils, file_utils
from pydoxtools import visual_document_analysis as vda
from pydoxtools import cluster_utils as gu
import pydoxtools as pdx

import pandas as pd
box_cols = gu.box_cols
from pydoxtools.settings import settings
import torch
import pathlib

from tqdm import tqdm

tqdm.pandas()

pdf_utils._set_log_levels()

nlp_utils.device, torch.cuda.is_available(), torch.__version__

# %% [markdown]
# ## load pdf files

# %%
# get all pdf files in subdirectory
pdf_files = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR / "pdfs", "*.pdf")
len(pdf_files)

# %% [markdown]
# ## try to extract table boundaries by indentifying lines
#
# 1. first we will group lines into boxes to identify "whole" cells the grouping was already provided by pdfminer.six --> **box_groups**
# 2. afterwards we will group those boxes into columns & rows --> **valid_columns**
# 3. as the third step, cluster the columns into tables --> table_groups

# %%
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/2-NeON2_V5_N1K_0LG_Datasheet_LGxxxN1K-V5_201905_EN.e5.pdf"
# pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/steca_PR_10-30_Datenblatt_DE.pdf"
# pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/POP300-Brochure.97.pdf"
# pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/POP300-Brochure.97.pdf"
#pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Horus.fb.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/42-SMA_SB3.0-6.0-1AV-41_PL_SMA_SB3-6-1VL-41_karta_katalogowa.6b.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/DS_CSK_PSPM_E_710-00711-A.60.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/ENP2019-086.B-IFM-Nano-Thruster-COTS-Product-Overview-1.ec.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Optocouplers.d3.pdf"
pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/6-Symo_M_klein_PL_Fronius_Symo_Karta_katalogowa.fe.pdf"
#pdf_file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/03-TP2M_DS_REC_TwinPeak_2_Mono_Series_IEC_Rev_D_ENG_WEB.0f.pdf"
training_data = pathlib.Path.home() / "comcharax/data"
pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
print(pdf_file)

# %% [markdown]
# todo:
#     
# - get max density in histogram and use that as a metric
# - get the variance of the histogram and use that as a metric...  tables should hae a pretty low
#   variance, as the elements are distributed more equally than with a figure

# %% [markdown]
# ## detect vertical & horizontal lines

# %% [markdown]
# ### initialize parameters

# %%
page=18
pdf = Document(pdf_file, page_numbers=[page])

# %%
hp = {'es1': 11.1, 'es2': 2.1, 'gs1': 11.1, 'gs2': 20.1}
adp = [{
    "va": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']],
    "ha": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']]
},
{
    "va": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']],
    "ha": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']]
}]
#pdfi = pdx.Document(
#    pdf_file,table_extraction_params=pydoxtools.extract_tables.TableExtractionParameters(
#        area_detection_distance_func_params =adp,
#        area_detection_threshold = 10.0
#    )

# %% [markdown]
# ## plot results

# %%
#pdf.elements

# %%
#pdf.table_df0[3]

# %%
# vda.plot_boxes(
#    df_le[vda.box_cols].values,
#    groups = df_le["hm"].values,
#    bbox = None, dpi=250)#p.page_bbox)

#print(pdf_file)
#for p in pdfi.pages[1:]:
boxes, box_levels = pdf.table_candidates, pdf.table_box_levels

# %%
vda.plot_box_layers(
    box_layers=[
        [pdf.graphic_elements[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red", filled=False)],
        [pdf.line_elements[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="blue")],
        [box_levels[18][0].values, vda.LayerProps(alpha=0.5, color="black", filled=False)],
        [box_levels[18][1].values if len(box_levels[18])>1 else [], vda.LayerProps(alpha=0.1, color="yellow", filled=True)],
        [pd.DataFrame([b._initial_area for b in boxes]).values, vda.LayerProps(alpha=1.0, color="red", filled=False, box_numbers=True)]
    ],
    bbox=pdf.pages_bbox[page], dpi=250
),

# %%
