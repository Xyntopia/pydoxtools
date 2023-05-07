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

dask.config.set(scheduler='threading')  # overwrite default with single-threaded scheduler for debugging

chroma_client = chromadb.Client()
project = chroma_client.create_collection(name="project")

# add database information
# connecting to postgres could be done like this:
# 'postgresql+psycopg2://user:password@hostname/db_name'
# it would also require installing the postgresql driver: pip install psycopg2
database_source = pydoxtools.document.DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="component_pages",  # select a table or use an actual SQL string
    index_column="id"
)

# oneliner to extract the information and form an index:
# create another DocumentBag using a subset of the extracted tabledata with the key "raw_html"
# and finally extract a dask-bag of dicts "get_dicts" from the DocumentBag
# which we will ingest into the vector store.
# or as a one-liner:
#   DocumentBag(source=database_source).get_data_docbag(column).get_dicts("source", "full_text", "vector")
table = DocumentBag(source=database_source).config(doc_configuration=dict(
    # here we can choose to do some fast vectorization by usong only the tokenizer
    vectorizer_only_tokenizer=True,
    vectorizer_model="sentence-transformers/all-MiniLM-L6-v2"
))
table.take(1)[0].data_sel("url","scrape_time")
d = table.e("data_sel","url").take(2)[0]

column = "url"  # which column(s) from the SQL table we want to extract
column = table.e("data_sel",column)  # multiple columns can be specified here.
column.take(1)

with ProgressBar():
    column.compute_index(100)  # choose a number how many rows you would like to add ro your chromadb!
    # run idx.map(add_to_chroma).persist()  to move all data into chromadb!

# query the db:
column.query_chroma("product")
