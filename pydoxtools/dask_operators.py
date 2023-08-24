from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import copy
from typing import Any
from typing import Callable

import dask.bag
from dask import dataframe
from dask.bag import Bag

from . import list_utils
from .document_base import Pipeline
from .operators_base import Operator, OperatorException


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

    def __call__(self, dask_bag: dask.bag.Bag, forgiving_extracts: bool, stats=None,
                 verbosity=None
                 ) -> Callable[[Any], dask.bag.Bag]:
        def forgiving_extract(doc: Pipeline, properties: list[str]) -> dict[str, Any]:
            # TODO: add option to this operator to use it as a "forgiving" extractor
            # TODO: it might be a good idea to move this directly into the pipeline as "forgiving_dict"?
            try:
                props = doc.to_dict(*properties)
                if stats is not None:
                    stats.append(copy.copy(doc._stats))
            except OperatorException:
                # we just continue  if an error happened. This is why we are "forgiving"
                props = {"Error": "OperatorException"}

            if len(properties) == 1:
                return props[properties[0]]
            else:
                return props

        def extract(doc: Pipeline, properties: list[str]) -> dict[str, Any]:
            props = doc.to_dict(*properties)
            if stats is not None:
                stats.append(copy.copy(doc._stats))
            if len(properties) == 1:
                return props[properties[0]]
            else:
                return props

        extract_func = forgiving_extract if forgiving_extracts else extract

        def safe_extract(*properties: list[str] | str) -> dask.bag.Bag:
            return dask_bag.map(extract_func, properties=properties)

        return safe_extract


class SQLTableLoader(Operator):
    """
    Load a table using dask/pandas read_sql

    sql: can either be the entire table or an SQL expression
    """

    def __call__(
            self, connection_string: str, sql: str, index_column: str,
            bytes_per_chunk: str
    ) -> dataframe.DataFrame:
        # tables = pd.read_sql("SELECT * FROM information_schema.tables", connection_string)
        # users = pd.read_sql("users", connection_string)
        # data = pd.read_sql(sql, connection_string)
        # TODO: read in a test row, find out what the index column is and use that automatically
        # engine = create_engine('mysql+pymysql://root:pass@localhost:3306/mydb')
        # query = 'SELECT * FROM my_table'

        # try:
        df = dataframe.read_sql_table(sql, connection_string, index_column, bytes_per_chunk=bytes_per_chunk)
        # except:
        #    df = pd.read_sql_query(sql=text(query), con=engine.connect())

        # npartitions=10, index_col='id'
        # we are not specifying the npartitions or divisions argument
        # so that dask automatically reduces the memory footprint for every partition
        # to about 256 MB.
        return df


def safe_mapping(func):
    def safe_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)

    return safe_func


class DocumentBagMap(Operator):
    """
    Basically it applies a function element-wise
    on documents in a dask bag and then creates a new DocumentBag from that. This
    works similar to pandas dataframes and series. But with documents
    as a basic datatype. And apply functions are also required to
    produce data which can be used as a document again (which is a lot).
    """

    def __call__(self, dask_bag: Bag, forgiving_extracts: bool,
                 stats=None, verbosity=None
                 ) -> Callable[..., dask.bag.Bag]:

        def document_mapping_with_stats(func):
            """gather statistics from a document mapping function"""

            def stats_mapping(d: Pipeline):
                res = func(d)
                if stats is not None:
                    stats.append(copy.copy(d._stats))
                return res

            return stats_mapping

        def extract_func_creator(props: list[str]):
            def extract(d):
                list_doc_mapping = list_utils.ensure_list(props)
                fobj = d.to_dict(*list_doc_mapping)
                fobj = list_utils.remove_list_from_lonely_object(fobj)
                return fobj

            return extract

        def mapping_creator(
                mapping_func: Callable | str | list[str]
        ) -> dask.bag.Bag:
            """
            will create a new document bag from an input documentbag, using a mapping function.

            The mapping functions has to accept a pydoxtools.Document as input.
            """
            if not callable(mapping_func):
                props = list_utils.ensure_list(mapping_func)
                mapping_func = extract_func_creator(props)
            if forgiving_extracts:
                mapping_func = safe_mapping(mapping_func)
            if stats is not None:
                mapping_func = document_mapping_with_stats(mapping_func)

            new_bag = dask_bag.map(mapping_func)
            return new_bag

        return mapping_creator
