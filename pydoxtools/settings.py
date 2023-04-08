#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:55:35 2020

general settings for comcharax project.
scrapy crawler specific settings can be found in
the the crawler directory and scrapy.cfg.

TODO:
- adapt this to kubernetes with environment variables or 
  docker parameters/environmen variables or whatever

@author: tom
"""

import logging
import os
from pathlib import Path

import appdirs
# TODO: remove joblib as a dependency from here ...
import joblib
from pydantic import BaseSettings

logger = logging.getLogger(__name__)

# from base64 import b64encode

_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_HOME = Path.home()

appname = "pydoxtools"
appauthor = "pydoxtools"


class _Settings(BaseSettings):
    CACHE_DIR_BASE: Path = Path(appdirs.user_cache_dir(appname, appauthor))
    TRAINING_DATA_DIR: Path = _FILE_DIR.parent / 'training_data'
    MODEL_DIR = CACHE_DIR_BASE / "models"

    # in order to be able to access OPENAI api
    OPENAI_API_KEY: str = "sk ...."

    """
    # model_type == "slow":
        # also very good, but slow:
        model_name = 'replydotai/albert-xxlarge-v1-finetuned-squad2'
    # model_type == "medium":
        # very good and fast:
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        # model_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'
    # model_type == "multi":
        # 'mrm8488/distilbert-multi-finedtuned-squad-pt'
        model_name = 'mrm8488/bert-multi-uncased-finetuned-xquadv1'
    # model_type == 'base':
        model_name = 'deepset/bert-base-cased-squad2'
    # model_type == 'large':
        model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
    # model_type == 't5':
        model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    # model_type == "fast":
        # not very good at this task, but fast
        # distilbert-base-cased-distilled-squad
        # model_name = 'mrm8488/bert-small-finetuned-squadv2'
        model_name = 'distilbert-base-cased-distilled-squad'
        # model_name = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    """
    PDXT_STANDARD_QAM_MODEL = 'distilbert-base-cased-distilled-squad'
    # be careful with this one here!!  we would have to retrain ALL of our
    # own, custom models!!!!
    PDXT_STANDARD_TOKENIZER = 'distilbert-base-multilingual-cased'

    # TODO: download classifiers in cache memory...
    def MODEL_STORE(self, name) -> Path:
        return self.MODEL_DIR / f"{name}classifier.ckpt"

    # TODO: replace this with diskcache
    def get_memory_cache(self):
        return joblib.Memory(str(self.CACHE_DIR_BASE), verbose=0)


settings = _Settings()
