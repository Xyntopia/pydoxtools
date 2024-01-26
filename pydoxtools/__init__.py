from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import importlib.metadata

__version__ = importlib.metadata.version("pydoxtools")

import logging

from .document import Document, DocumentBag, DatabaseSource
from .document_base import Pipeline

logger = logging.getLogger(__name__)
