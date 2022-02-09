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
from secrets import token_hex

import appdirs
# TODO: remove joblib as a dependency from here ...
import joblib
from pydantic import BaseSettings, DirectoryPath

logger = logging.getLogger(__name__)

# from base64 import b64encode

_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_SOURCE_DIR = Path(_FILE_DIR).parent.parent
_HOME = Path.home()

appname = "pydoxtools"
appauthor = "pydoxtools"

class _Settings(BaseSettings):
    CACHE_DIR_BASE: Path = appdirs.user_cache_dir(appname, appauthor)
    TRAINING_DATA_DIR: Path = _FILE_DIR.parent / 'training_data'

    @property
    def MODELDIR(self) -> Path:
        return self.APPDATA / "models"

    # TODO: download classifiers in cache memory...
    def CLASSIFIER_STORE(self, name) -> Path:
        return self.MODELDIR / f"{name}classifier.ckpt"

    @property
    def classifier_model_file_gpu(self) -> Path:
        return self.MODELDIR / "page_classifier_pipeline.gpu.pickle"

    def get_memory_cache(self):
        return joblib.Memory(str(self.CACHE_DIR_BASE), verbose=0)


settings = _Settings()
