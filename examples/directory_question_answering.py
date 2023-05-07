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

d = DocumentBag(['../README.md','../DEVELOPMENT.md'])
d = DocumentBag(['../README.md'])
d.bag.take(1)
d.file_path_list.take(1)
d.dir_list.take(1)
d.docs.take(1)[0].full_text
d.pipeline_chooser

d.d.full_text
d.d.text_segments
d.sub_doc("text_segments")
d.to_dict("text_segments")
[len(t) for t in d.text_segments]
