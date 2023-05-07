"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import dask.bag

from pydoxtools import DocumentBag

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

root_dir = "../../pydoxtools"
ds = DocumentBag(
    source=root_dir,
    exclude=[
        '.git/', '.idea/', '/node_modules', '/dist',
        '/__pycache__/', '.pytest_cache/'
    ])

# ds.take(10)

ds = DocumentBag(['../README.md','../DEVELOPMENT.md'])
#ds = DocumentBag(['../README.md'])
#ds.bag.take(1)
#ds.file_path_list.take(1)
#ds.dir_list.take(1)
#ds.docs.take(1)[0].full_text
ds.pipeline_chooser
ds.docs.take(10)
ds.e('text_segments').docs#.flatten().compute()
ds.e('text_segments').compute()

d = ds.docs.take(1)[0]
getattr(d,"document_type")
#d['embedding']
#d.x('embeddings')
#d.get('embedding')
