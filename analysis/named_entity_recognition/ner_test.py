# # Analyze the extraction of a single component from pdf

# +
import sys

# %load_ext autoreload
# %autoreload 2

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

# +
from pathlib import Path

from pydoxtools import nlp_utils, load_document
from pydoxtools import pdf_utils, file_utils
from pydoxtools.settings import settings
import torch
import random
from operator import attrgetter

import pandas as pd
from tqdm import tqdm
from IPython.display import display, Markdown, Latex
import os
from os.path import join

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__

TRAINING_DIR = Path("../../../training_data/")
# -

# ## analyze sample html + optional pdf

# +
# get all pdf files in subdirectory

files = file_utils.get_nested_paths(TRAINING_DIR / "pdfs/datasheet", "*.pdf")
len(files)
# -

pdf_file = TRAINING_DIR / "pdfs/datasheet/LG-Datenblatt_NeON2_V5_2019_DE.b6.pdf"
pdf_file=random.choice(files)
print(pdf_file)

pdf = load_document(pdf_file)
pdf

# d = pdf_utils.extract_pdf_data(str(pdf_file))
with open(pdf_file, "rb") as pf:
    doc = load_document(pf, source="jupyter_lab")
    d = doc.model

len(d.tables)
d.tables[0]

doc.urls

txt = doc.full_text
#doc.textboxes

# +
#nlp_utils.init_nlp_cache()
# -

print(txt)

# +
# #!python -m spacy download xx_ent_wiki_sm
# -

doc.ner

# +
from transformers import pipeline

#ner_pipe = pipeline("ner")
#good results = "xlm-roberta-large-finetuned-conll03-english" # large but good
#name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english" #small and bad 
name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
model = name
tokenizer= name
ner_pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)
# -

ner_pipe(txt)


