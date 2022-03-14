# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
from pathlib import Path
import subprocess
import pydoxtools
import pytesseract
from PIL import Image

# %%
file = Path("/home/tom/Sync/powerbox/documents/20211215 - documents/IMG_20211221_145240.jpg")

# %%
file

# %%
#print(pytesseract.image_to_string(Image.open(file), lang='deu'))
pdf = pytesseract.image_to_pdf_or_hocr(Image.open(file), extension='pdf')

# %%
import io
doc = pydoxtools.load_document(io.BytesIO(pdf))

# %%
doc = pydoxtools.load_document(file, ocr=True, ocr_lang="deu+eng")

# %%
print(doc.full_text)

# %%
