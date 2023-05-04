from typing import Any, Callable

import dask.bag
from dask import dataframe

from pydoxtools.operators_base import forgiving_extract
from pydoxtools.document_base import Operator


class BagMapOperator(Operator):
    """Applies any function on items in a dask bag"""

    def __init__(self, func: callable):
        super().__init__()
        self._func = func

    def __call__(self, dask_bag: dask.bag.Bag) -> dask.bag.Bag:
        return dask_bag.map(lambda item: self._func(item))


class BagPropertyExtractor(Operator):
    """
    Returns a function closure which returns a bag of the specified
    property of the enclosed documents.
    """

    def __call__(self, dask_bag: dask.bag.Bag) -> Callable[[Any], dask.bag.Bag]:
        def safe_extract(properties: list[str] | str) -> dask.bag.Bag:
            return dask_bag.map(forgiving_extract, properties=properties)

        return safe_extract

class SQLTableLoader(Operator):
    """
    Load a table using dask/pandas read_sql
    """

    def __call__(self, connection_string: str, sql: str, index_column: str) -> dataframe:
        # tables = pd.read_sql("SELECT * FROM information_schema.tables", connection_string)
        # users = pd.read_sql("users", connection_string)
        # data = pd.read_sql(sql, connection_string)
        # TODO: read in a test row, find out what the index column is and use that automatically
        df = dataframe.read_sql_table(sql, connection_string, index_column)
        # npartitions=10, index_col='id'
        # we are not specifying the npartitions or divisions argument
        # so that dask automatically reduces the memory footprint for every partition
        # to about 256 MB.
        return df
