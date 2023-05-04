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
database_source = pydoxtools.document.DatabaseSource(
    connection_string="sqlite:///" + str(Path.home() / "comcharax/data/component_pages.db"),
    sql="component_pages",  # simply select a table for example
    index_column="id"
)

# oneliner to extract the information and form an index:
idx = DocumentBag(source=database_source). \
    get_data_docbag("raw_html"). \
    get_dicts("source", "full_text", "vector")


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
