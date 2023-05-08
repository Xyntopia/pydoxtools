from typing import Any
from typing import Callable

import dask.bag
from dask import dataframe

from pydoxtools.document_base import Operator, OperatorException, Pipeline


class BagMapOperator(Operator):
    """Applies any function on items in a dask bag"""

    def __init__(self, func: callable):
        super().__init__()
        self._func = func

    def __call__(self, dask_bag: dask.bag.Bag, *args, **kwargs) -> dask.bag.Bag:
        return dask_bag.map(self._func, *args, **kwargs)


class BagFilterOperator(Operator):
    """Applies any function on items in a dask bag and filters them based on the result.
    if func returns False, the element will be dropped from the bag."""

    def __init__(self, func: callable):
        super().__init__()
        self._func = func

    def __call__(self, dask_bag: dask.bag.Bag) -> dask.bag.Bag:
        return dask_bag.filter(self._func)


class BagPropertyExtractor(Operator):
    """
    Returns a function closure which returns a bag of the specified
    property of the enclosed documents.
    """

    def __call__(self, dask_bag: dask.bag.Bag) -> Callable[[Any], dask.bag.Bag]:
        def forgiving_extract(doc: Pipeline, properties: list[str]) -> dict[str, Any]:
            # TODO: add option to this operator to use it as a "forgiving" extractor
            # TODO: it might be a good idea to move this directly into the pipeline as "forgiving_dict"?
            try:
                props = doc.to_dict(*properties)
            except OperatorException:
                # we just continue  if an error happened. This is why we are "forgiving"
                props = {"Error": "OperatorException"}

            if len(properties) == 1:
                return props[properties[0]]
            else:
                return props

        def extract(doc: Pipeline, properties: list[str]) -> dict[str, Any]:
            props = doc.to_dict(*properties)

            if len(properties) == 1:
                return props[properties[0]]
            else:
                return props

        def safe_extract(*properties: list[str] | str) -> dask.bag.Bag:
            return dask_bag.map(extract, properties=properties)

        return safe_extract


class SQLTableLoader(Operator):
    """
    Load a table using dask/pandas read_sql
    """

    def __call__(
            self, connection_string: str, sql: str, index_column: str,
            bytes_per_chunk: str
    ) -> dataframe.DataFrame:
        # tables = pd.read_sql("SELECT * FROM information_schema.tables", connection_string)
        # users = pd.read_sql("users", connection_string)
        # data = pd.read_sql(sql, connection_string)
        # TODO: read in a test row, find out what the index column is and use that automatically
        df = dataframe.read_sql_table(sql, connection_string, index_column, bytes_per_chunk=bytes_per_chunk)
        # npartitions=10, index_col='id'
        # we are not specifying the npartitions or divisions argument
        # so that dask automatically reduces the memory footprint for every partition
        # to about 256 MB.
        return df
