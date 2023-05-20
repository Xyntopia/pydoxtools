"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

import logging
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dask.diagnostics import ProgressBar

import pydoxtools
from pydoxtools import DocumentBag
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # in order to "scale up" the operation, we can use dask distributed.
    # As pydoxtools.Document is pickable, this is possible..
    import dask

    # dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    # need to be inside the "__main__".
    from dask.distributed import Client
    client = Client()  # set up local cluster on your laptop

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

    # In this apply operation we choose the information that we want from
    # each row in the table (represented by a pydoxtools.Document).
    # Document.apply automatically create a new list documents of it and
    # returns that as a new DocumentBag
    db_idx = table.apply(new_document=lambda d: [d.data["url"], d.data["scrape_time"]])

    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(settings.PDX_CACHE_DIR_BASE / "chromadb")
    )
    collection_name = "index"


    # we can now add the contents of our index-DocumentBagto chromadb
    # we can either do it manually:
    def insert(docs):
        chroma_client = chromadb.Client(chroma_settings)
        collection = chroma_client.get_or_create_collection(name=collection_name)

        for i, doc in enumerate(docs):
            collection.add(
                embeddings=[[float(n) for n in doc.embedding]],
                documents=[doc.full_text],
                metadatas=[{"source": str(doc.source)}],
                ids=[uuid.uuid4().hex]
            )

        chroma_client.persist()

        return range(i)


    # or let pydoxtools do the boilerplate, which
    # also gives us a query function to automatically convert the vectors:
    compute, query = db_idx.add_to_chroma(
        chroma_settings, collection_name,
        # embeddings="embedding", # we can customize which embeddings to take here, default is "embedding" of the full text
        # document="full_text", # we can customize which field to take here, default is "full_text"
        metadata="source",  # this is also customizable, the default would simply take the "metadata" of each document
        # ids=None  # we can also create ids from the document for example using the "source" as id or similar
    )

    manual = False
    with ProgressBar():
        if manual:
            a = db_idx.docs.map_partitions(insert).take(20)
        else:
            compute(100)  # choose a number how many rows you would like to add ro your chromadb!

    res = query("raspberry pi products")

    # query the db:
    # column.query_chroma("product")
