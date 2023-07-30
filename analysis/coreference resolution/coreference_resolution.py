# # Play around with rule-based coreference resolution strategies

# +

# %load_ext autoreload
# %autoreload 2

import logging
from pathlib import Path

import pandas as pd
import spacy
import torch
from tqdm import tqdm

from pydoxtools import nlp_utils, Document
from pydoxtools import pdf_utils, file_utils
from pydoxtools.settings import settings

import pydoxtools.visualization as vd
import html

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

tqdm.pandas()

pdf_utils._set_log_levels()

nlp_utils.device, torch.cuda.is_available(), torch.__version__
# -

# ## analyze sample html + optional pdf

files = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
len(files)

pdf_file = settings.TRAINING_DATA_DIR / "pdfs/whitepaper/En-Sci-Application-Brief.88.pdf"
# pdf_file=random.choice(files)
print(pdf_file.absolute())

# get all pdf files in subdirectory
training_data_dir = 0
pdf_file = Path("../../README.md")

pdf = Document(pdf_file, spacy_model_size="sm")
doc = pdf

doc.configuration

relationships=doc.relationships
text=doc.spacy_doc.text
G = doc.knowledge_graph
#G=KG

vd.draw(G, format='svg')

# +
#dotgraph.write()

# +
#print(str(dotgraph)[:1000])
# -

if False:
    line = 1429
    context = 10
    for i,l in enumerate(str(dotgraph).split("\n")[line-context:line+context]):
        print(f"{i+line-context}: {l}")
