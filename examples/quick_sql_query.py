"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

from pathlib import Path

import dask

from pydoxtools import DocumentBag, DatabaseSource

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

# add database information
database_source = DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="component_pages",  # simply select a table for example
    index_column="id"
)

# Extract the information and form an index:
# create another DocumentBag using a subset of the extracted tabledata with the key "raw_html"
# and finally extract a dask-bag of dicts "get_dicts" from the DocumentBag
# which we will ingest into the vector store.
docs = DocumentBag(source=database_source).config(doc_configuration=dict(
    # here we can choose to do some fast vectorization by usong only the tokenizer
    vectorizer_only_tokenizer=False,
    vectorizer_model="sentence-transformers/all-MiniLM-L6-v2"
))
column = docs.e('data_sel', "url")
idx = column.compute_index(100)
res = column.query_chroma("product")  # get URLs which are related to "products":
