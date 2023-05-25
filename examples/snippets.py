import dask.config
import pydoxtools as pdx

# overwrite default with single-threaded scheduler for debugging and so that we don#t need
# a main function...
dask.config.set(scheduler='synchronous')

docs = pdx.DocumentBag("../tests/data", forgiving_extracts=True)
res = docs.bag_apply(["tables_df", "filename"]).compute()
