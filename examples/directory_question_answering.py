"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import dask.bag

from dask.diagnostics import ProgressBar
from pydoxtools import DocumentBag

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

root_dir = "../../pydoxtools"
ds = DocumentBag(
    source=root_dir,
    exclude=[
        '.git/', '.idea/', '/node_modules', '/dist',
        '/__pycache__/', '.pytest_cache/'
    ])

#ds = DocumentBag(['../README.md', '../DEVELOPMENT.md'])

idx = ds.e('text_segments')

with ProgressBar():
    idx.compute_index(10) # remove number to calculate for all files!

idx.query_chroma("How to contribute to this library?")
