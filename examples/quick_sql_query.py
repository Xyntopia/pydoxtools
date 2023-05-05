"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

from pathlib import Path

import dask

import pydoxtools
from pydoxtools import DocumentBag, Document

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

# add database information
database_source = pydoxtools.document.DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="component_pages",  # simply select a table for example
    index_column="id"
)

# oneliner to extract the information and form an index:
# create another DocumentBag using a subset of the extracted tabledata with the key "raw_html"
# and finally extract a dask-bag of dicts "get_dicts" from the DocumentBag
# which we will ingest into the vector store.
idx = DocumentBag(source=database_source).get_data_docbag("raw_html")
idx.take(2)
idx