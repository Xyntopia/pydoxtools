import importlib.metadata

__version__ = importlib.metadata.version("pydoxtools")

import logging

from .document import Document, DocumentBag, DatabaseSource
from .document_base import Pipeline
from .agent import LLMAgent

logger = logging.getLogger(__name__)
