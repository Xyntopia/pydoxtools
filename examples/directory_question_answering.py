"""Demonstrate how to load a directory into an index and
then answer questions about it"""

from pydoxtools import DocumentBag, Document

import dask.bag
dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

root_dir = "../../pydoxtools"
ds = DocumentBag(
    source=root_dir,
    exclude=[
        '.git/', '.idea/', '/node_modules', '/dist',
        '/__pycache__/', '.pytest_cache/'
    ])

ds.take(10)
