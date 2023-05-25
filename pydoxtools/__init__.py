__version__ = '0.5.1'

import logging

from .document import Document, DocumentBag, DatabaseSource
from .document_base import Pipeline
from .agent import Agent

logger = logging.getLogger(__name__)
