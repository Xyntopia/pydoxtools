"""
In this example we will extract information from
an SQL database and inject it into chroma db.
"""

import logging
from pathlib import Path

import chromadb
import dask

import pydoxtools
from pydoxtools import DocumentBag

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
table = DocumentBag(source=database_source).config(
    doc_configuration=dict(
        # here we can choose to do some fast vectorization by usong only the tokenizer
        vectorizer_only_tokenizer=True,
        vectorizer_model="sentence-transformers/all-MiniLM-L6-v2", ),
    forgiving_extracts=True,
)

# demonstrate different ways on how we can extract data from documents:
# table.take(5)[0].data.keys()
# table.take(5)[0].data["url"]
columns = "url", "scrape_time"  # which column(s) from the SQL table we want to extract
# column = table.e(columns, meta_properties=["index"])  # multiple columns can be specified here.
a = table.d("data").take(5)
b = table.bag_apply(lambda d: [d.data["url"], d.data["index"]]).take(5)
b = table.apply(
    lambda d: dict(url=d.data["url"], ind=d.data["index"]),
    lambda d: {"index": str(d.data["index"]) + d.data["url"]}
).take(5)

c = table.e(["url", "index"]).d("data").take(5)  # automatically create metadata for new documents
d = table.e("url", meta_properties=["index"]).d("data").take(5)
d[0]._fobj  # should be an URL
# column.take(5)

# query the db:
# column.query_chroma("product")
