# # Play around with rule-based coreference resolution strategies

# +

# %load_ext autoreload
# %autoreload 2

import logging
from operator import attrgetter

import componardo.visualization
import hnswlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from IPython.core.display import display
from IPython.display import HTML
from pydoxtools import nlp_utils, load_document
from pydoxtools import pdf_utils, file_utils, cluster_utils as cu
from pydoxtools.settings import settings
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__
# -

# ## analyze sample html + optional pdf

# +
# get all pdf files in subdirectory

files = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
len(files)
# -

pdf_file = settings.TRAINING_DATA_DIR / "pdfs/whitepaper/En-Sci-Application-Brief.88.pdf"
# pdf_file=random.choice(files)
print(pdf_file.absolute())

pdf = load_document(pdf_file, model_size="trf")
doc = pdf

# try out coreference based on nearest neighbours..
sent = list(doc.spacy_doc.sents)[10]

[(t.text,t.pos_) for t in sent]

w = sent[0]
w.morph

sent = list(doc.spacy_doc.sents)[8]
sent

pd.DataFrame([(t.text, t.pos_,spacy.explain(t.pos_),t.tag_,spacy.explain(t.tag_), t.morph) for t in sent])

print(sent[11])
[(r,r[0].sent) for r in doc.knn_query(sent[11], k=10, pos_filter="PRON")]

pron = sent[1]
pron

# search for all words...
doc.knn_query(pron, k=10)

pd.DataFrame(list((s.text,s) for s in doc.spacy_doc.sents))

doc.knn_query(sent[0], k=50, pos_filter="NOUN")


