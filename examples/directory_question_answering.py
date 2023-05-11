"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import dask.bag
from dask.diagnostics import ProgressBar

from pydoxtools import DocumentBag
from pydoxtools.settings import settings

#settings.PDX_ENABLE_DISK_CACHE = True
dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

root_dir = "../../pydoxtools"
ds = DocumentBag(
    source=root_dir,
    exclude=[
        '.git/', '.idea/', '/node_modules', '/dist',
        '/__pycache__/', '.pytest_cache/', '.chroma'
    ])

ds = DocumentBag(['../.git', '../README.md', '../DEVELOPMENT.md'])
ds.stats

idx = ds.e('text_segments')
idx.stats
# d1 = idx.take(10)[0]
# d2 = idx.take(10)[1]


with ProgressBar():
    idx.compute_index()  # remove number to calculate for all files!

idx.docs.count().compute()


idx.query_chroma("How to contribute to this library?")
