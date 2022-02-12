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

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
import pydoxtools
from pydoxtools import pdf_utils, classifier, nlp_utils
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings
import torch
from IPython.display import display
import re
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
# # translate text boxes into vectors...

# %% [markdown]
# TODO: its probabybl a ood idea to use some hyperparemeter optimization in order to find out what is the best method here...
#
# we would probably need some manually labeled addresses from our dataset for this...

# %% [markdown]
# training...

# %%
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/Datenblatt_PSL-Family.37.pdf"
file = settings.TRAINING_DATA_DIR / "pdfs/datasheet/remo-m_fixed-wing.2f.pdf"

# %%
file

# %%
model = classifier.load_classifier("text_block")

# %%
txtblocks = pd.DataFrame(pydoxtools.load_document(file).textboxes)
pred = txtblocks.txt.apply(lambda x: model.predict_proba([x])[0])
txtblocks[["add_prob", "ukn_prob"]] = pred.apply(pd.Series).round(2)
pretty_print(txtblocks)

# %%
example_address = ["""
Leptron Unmanned Aircraft Systems, Inc.
2650 East 40th Avenue
Denver, Colorado 80205
Phone 303-384-3469
FAX 303-322-7242
www.leptron.com
""",
"""
XCAM Ltd.
2 Stone Circle Road
Northampton
NN3 8RF
UK
""",
"""
Leptron Unmanned Aircraft Systems, Inc. </s>1-800-722-2800
2650 East 40th Avenue
Denver, Colorado 80205
Phone 303-384-3469
FAX 303-322-7242
www.leptron.com
""",
"""
WORLD HEADQUARTERS: 
233 Kansas St., 
El Segundo, California 90245, USA Tel: (310) 252-7105\nTAC Fax: (310) 252-7903
""",
"""
WORLD HEADQUARTERS: 
233 Kansas St., El Segundo, California 90245, Tel: (310) 252-7105
""",
"""
233 Kansas St.
El Segundo
California 90245, USA 
TAC Fax: (310) 252-7903
""","""
Flt No 340R
126 Woodfield Avenue
Colinton
Edinburgh United Kingdom
""","""
ex king ltd
Springfield Gardens
Queens
N. Y 11413
""","""
ex king ltd
Springfield Gardens
Queens
N. Y 11413
http://www.something.com
""",
"""
ex king ltd
Springfield Gardens
Queens
N. Y 11413
www.something.com
""",
"""
11711 N Creek Pkwy S., Suite D-113, Bothell, WA 98011 •\xa0425-486-0100
""",
"""
An die  \naustro mechana GmbH \nBaumannstrasse 10 \nA-1030 Wien  \nFax: +43 (0)1 717 87 778 \nsmv@akm.at
""",
"""
Murata Power Solutions, Inc. \n129 Flanders Road, Westborough, MA 01581 U.S.A. \nISO 9001 and 14001 REGISTERED
""",
"""
Postal address: VeraSafe United Kingdom, Ltd., 37 Albert Embankment, London, SE1 7TL, United 
""",
"""
Tell: +86-020-36380552, +86-020-36042809\nhttp://www.aosong.com\nEmail: thomasliu198518@yahoo.com.cn\nAddress: No.56, Renhe Road, Renhe Town, Baiyun District, Guangzhou, China',
       'Aosong(Guangzhou) Electronics Co.,Ltd\n
""",
"""
One Technology Way, P.O. Box 9106, Norwood, MA 02062-9106, U.S.A.
Tel: 781/329-4700\nwww.analog.com\n© Analog Devices, Inc., 2002\nFax: 781/326-8703
""",
"""
Fluke Deutschland GmbH\nIn den Engematten 14\n79286 Glottertal\nTelefon: (069) 2 22 22 02 00 \nTelefax: (069) 2 22 22 02 01\nE-Mail: info@de.ﬂ uke.nl\nWeb: www.ﬂ uke.de
""",
"""
Fluke Deutschland GmbH\nIn den Engematten 14
79286 Glottertal
Telefon: (069) 2 22 22 02 00 
Telefax: (069) 2 22 22 02 01
E-Mail: info@de.ﬂ uke.nl
Web: www.ﬂ uke.de
""",
"""
CS Wismar GmbH · An der Westtangente 1 · 23966 Wismar · Germany · +49 38 413 049 300
""",
"""
CS Wismar GmbH  An der Westtangente 1  23966 Wismar  Germany  +49 38 413 049 300
""",
                   """
JA SOLAR Co., Ltd
Add:5th Jionglong Industrial Park</s>.<s>No.123 Xinxing Road
Ningjin, Xingtai City, Hebei Province, China
Tel:+86 03195808112\nFax:+86 03195808112
""",
                                      """
JA SOLAR Co., Ltd
Add:5th Jionglong Industrial Park. No.123 Xinxing Road
Ningjin, Xingtai City, Hebei Province, China
Tel:+86 03195808112\nFax:+86 03195808112
""",
"""
F U R T H E R   I N F O R M AT I O N  Max-Planck-Str. 3    |    12489 Berlin, Germany  |  info@berlin-space-tech.com  |   www.berlin-space-tech.com
""",
"""
Delta Electronics (Netherlands) B.V. 
Zandsteen 15, 2132 MZ HOOFDDORP, Niederlande,  
\uf027 : T +31(0)20 800 3900: +31(0)20 8003999, \uf02a : info@deltaww.com   \uf03a : www.delta-emea.com 
KvK-Nummer: 12040831, USt-IDNr.: NL 8085.73.986.B.01,  
Bankverbindung: Citibank International plc, Netherlands Branch, EURO Kontonummer: 26.60.61.095 IBAN: NL52 CITI 0266 0610 95, \nUSD Kontonummer: 10.63.77.812  IBAN: NL65 CITI 0106 3778 12
""",
"""
Delta Electronics (Netherlands) B.V. 
Zandsteen 15, 2132 MZ HOOFDDORP, Niederlande,  
\uf027 : T +31(0)20 800 3900: +31(0)20 8003999, \uf02a : info@deltaww.com   \uf03a : www.delta-emea.com 
""",
"""
Astro- und Feinwerktechnik Adlershof GmbH \nAlbert-Einstein-Str. 12 \nD- 12489 Berlin
""",
"""
dtL，.oC ygolonhceT VP)iefeH（ RALOS AJ
Add:No. 999 Changning Avenue, Gaoxin District, 
Hefei City, Anhui Province, China
Tel:+86(551)530 5525
Fax:+86(10)530 5533
"""
                  ]
model.predict_proba(example_address).numpy().round(3)

# %%
