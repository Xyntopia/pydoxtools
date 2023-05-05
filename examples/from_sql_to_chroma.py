"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

from pathlib import Path

import chromadb
import dask

import pydoxtools
from pydoxtools import DocumentBag, Document

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

chroma_client = chromadb.Client()
project = chroma_client.create_collection(name="project")

# add database information
# connecting to postgres could be done like this:
# 'postgresql+psycopg2://user:password@hostname/db_name'
# it would also require installing the postgresql driver: pip install psycopg2
database_source = pydoxtools.document.DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="SELECT * FROM component_pages",  # select a table or use an actual SQL string
    index_column="id"
)

# oneliner to extract the information and form an index:
# create another DocumentBag using a subset of the extracted tabledata with the key "raw_html"
# and finally extract a dask-bag of dicts "get_dicts" from the DocumentBag
# which we will ingest into the vector store.
# or as a one-liner:
#   DocumentBag(source=database_source).get_data_docbag(column).get_dicts("source", "full_text", "vector")
table = DocumentBag(source=database_source)
column = "raw_html"  # which column(s) from the SQL table we want to extract
column = table.get_data_docbag(column) # multiple columns can be specified here.
# we create the index by creating a "vector" out of the selected column/s
idx = column.get_dicts("source", "full_text", "vector")

# we add our data to chroma line-by-line. There is not much overhead involved in this
# as the most expensive operation here is the calculation for the embeddings
def add_to_chroma(item: dict):
    project.add(
        # TODO: embeddings=  #use our own embeddings for specific purposes...
        embeddings=[[float(n) for n in item["vector"]]],
        documents=[item["full_text"]],
        metadatas=[{"source": item["source"]}],
        ids=[item["source"]]
    )

idx.map(add_to_chroma).take(20)  # choose a number how many rows you would like to add ro your chromadb!

# query the db:
project.query(Document("db query").vector.tolist())
