"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

import logging
from pathlib import Path

import chromadb
import dask
from dask.diagnostics import ProgressBar

import pydoxtools
from pydoxtools import DocumentBag, Document

logger = logging.getLogger(__name__)

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="index")

# add database information
# connecting to postgres could be done like this:
# 'postgresql+psycopg2://user:password@hostname/db_name'
# it would also require installing the postgresql driver: pip install psycopg2
database_source = pydoxtools.document.DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="component_pages",  # select a table or use an actual SQL string
    index_column="id"
)

# Extract the information and form an index:
# create another DocumentBag using a subset of the extracted tabledata with the key "raw_html"
# and finally extract a dask-bag of dicts "get_dicts" from the DocumentBag
# which we will ingest into the vector store.
# or as a one-liner:
#   DocumentBag(source=database_source).get_data_docbag(column).get_dicts("source", "full_text", "vector")
# this will load each table row into a single row as a "dictionary"
table = DocumentBag(source=database_source).config(doc_configuration=dict(
    # here we can choose to do some fast vectorization by usong only the tokenizer
    vectorizer_only_tokenizer=True,
    vectorizer_model="sentence-transformers/all-MiniLM-L6-v2"
))

compute, query = table.apply(
    new_document=lambda d: [d.data["url"], d.data["scrape_time"], ]
).add_to_chroma(
    collection,
    # embeddings="embedding", # we can customize which embeddings to take here, default is "embedding" of the full text
    # document="full_text", # we can customize which field to take here, default is "full_text"
    metadata="source",  # this is also customizable, the default would simply take the "metadata" of each document
    ids=None  # we can also create ids from the document for example using the "source" as id or similar
)

with ProgressBar():
    compute(20)  # choose a number how many rows you would like to add ro your chromadb!

res = query(embeddings=Document("list").embedding.tolist())
res = query("raspberry pi products")

# query the db:
# column.query_chroma("product")
