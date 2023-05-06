"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

from pathlib import Path

import dask
from dask.diagnostics import ProgressBar

import pydoxtools
from pydoxtools import DocumentBag

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
docs = DocumentBag(source=database_source).config(doc_configuration=dict(
    # here we can choose to do some fast vectorization by usong only the tokenizer
    vectorizer_only_tokenizer=True,
    vectorizer_model="sentence-transformers/all-MiniLM-L6-v2"
))
d = docs.take(1)[0]
column = docs.get_data_docbag("raw_html")
docs.get_datadocs("url").take(1)[0].configuration
c = column.take(1)[0]
a = column.get_dicts("embedding").take(2)
idx = column.get_dicts("source", "full_text", "embedding")

with ProgressBar():
    b = idx.take(20)
