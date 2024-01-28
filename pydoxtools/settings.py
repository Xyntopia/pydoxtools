#!/usr/bin/env python3
"""
Created on Tue Apr 28 21:55:35 2020

@author: tom
"""

from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import os
from pathlib import Path

import appdirs
# TODO: remove joblib as a dependency from here ...
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# from base64 import b64encode

_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_PYDOXTOOLS_DIR = _FILE_DIR.parent
_HOME = Path.home()

appname = "pydoxtools"
appauthor = "pydoxtools"


class _Settings(BaseSettings):
    PDX_CACHE_DIR_BASE: Path = Path(appdirs.user_cache_dir(appname, appauthor))
    PDX_ENABLE_DISK_CACHE: bool = False
    TRAINING_DATA_DIR: Path = _PYDOXTOOLS_DIR / 'training_data'
    PDX_MODEL_DIR: Path = PDX_CACHE_DIR_BASE / "models"

    # in order to be able to access OPENAI api
    OPENAI_API_KEY: str = "sk ...."

    # PDXT_STANDARD_QAM_MODEL = 'distilbert-base-cased-distilled-squad'
    PDXT_STANDARD_QAM_MODEL: str = 'deepset/minilm-uncased-squad2'
    # be careful with this one here!!  we would have to retrain ALL of our
    # own, custom models!!!!
    PDXT_STANDARD_TOKENIZER: str = 'distilbert-base-multilingual-cased'

    # TODO: download classifiers in cache memory...
    def MODEL_STORE(self, name) -> Path:
        return self.PDX_MODEL_DIR / f"{name}classifier.ckpt"


settings = _Settings()
