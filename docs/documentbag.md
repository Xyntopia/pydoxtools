# [pydoxtools.DocumentBag][]

::: pydoxtools.DocumentBag

## Text extraction attributes and functions

The [pydoxtools.DocumentBag][] is built
on the [pydoxtools.Pipeline][] class and most of the text extraction
functionality makes extensive use of the pipeline features. All attributes
and functions that are created by the pipeline are documented here.

Pipeline visualizations for the structure of the Document pipelines for different
document types can be found [here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

### Document

Get a factory for pre-configured documents. Can be called just like [pydoxtools.Document][] class, but automatically gets assigned the same configuration as all Documents in this bag

*name*
: `<DocumentBag>.x('Document') or <DocumentBag>.Document`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### _stats

A constant value

*name*
: `<DocumentBag>.x('_stats') or <DocumentBag>._stats`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### apply

Basically it creates a Documentbag from two sets of
    on documents in a dask bag and then creates a new DocumentBag from that. This
    works similar to pandas dataframes and series. But with documents
    as a basic datatype. And apply functions are also required to
    produce data which can be used as a document again (which is a lot).

*name*
: `<DocumentBag>.x('apply') or <DocumentBag>.apply`

*return type*
: typing.Callable[..., pydoxtools.document.DocumentBag]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### bag

| pipe_type                                                    | description                                    |
|:-------------------------------------------------------------|:-----------------------------------------------|
| <class 'list'\>, <class 'pydoxtools.document.DatabaseSource'\> | No documentation                               |
| <class 'pathlib.Path'\>                                       | create a dask bag with all the filepaths in it |

*name*
: `<DocumentBag>.x('bag') or <DocumentBag>.bag`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### bag_apply

Basically it applies a function element-wise
    on documents in a dask bag and then creates a new DocumentBag from that. This
    works similar to pandas dataframes and series. But with documents
    as a basic datatype. And apply functions are also required to
    produce data which can be used as a document again (which is a lot).

*name*
: `<DocumentBag>.x('bag_apply') or <DocumentBag>.bag_apply`

*return type*
: typing.Callable[..., dask.bag.core.Bag]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### compute

No documentation

*name*
: `<DocumentBag>.x('compute') or <DocumentBag>.compute`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### connection_string

No documentation

*name*
: `<DocumentBag>.x('connection_string') or <DocumentBag>.connection_string`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'pydoxtools.document.DatabaseSource'\>

### d

Alias for: 

\* get_dicts-\>d (output)

*name*
: `<DocumentBag>.x('d') or <DocumentBag>.d`

*return type*
: typing.Callable[[typing.Any], dask.bag.core.Bag]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### dataframe

Load a table using dask/pandas read_sql

    sql: can either be the entire table or an SQL expression

*name*
: `<DocumentBag>.x('dataframe') or <DocumentBag>.dataframe`

*return type*
: <class 'dask.dataframe.core.DataFrame'\>

*supports pipeline flows*
: <class 'pydoxtools.document.DatabaseSource'\>

### dir_list

| pipe_type              | description                                                                       |
|:-----------------------|:----------------------------------------------------------------------------------|
| <class 'list'\>         | Applies any function on items in a dask bag and filters them based on the result. |
|                        |     if func returns False, the element will be dropped from the bag.              |
| <class 'pathlib.Path'\> | No documentation                                                                  |

*name*
: `<DocumentBag>.x('dir_list') or <DocumentBag>.dir_list`

*return type*
: <class 'dask.bag.core.Bag'\> | typing.Any

*supports pipeline flows*
: <class 'list'\>, <class 'pathlib.Path'\>

### docs

| pipe_type                                    | description                                                                                                                |
|:---------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| <class 'dask.bag.core.Bag'\>                  | Alias for:                                                                                                                 |
|                                              |                                                                                                                            |
|                                              | \* source-\>docs (output)                                                                                                    |
| <class 'pydoxtools.document.DatabaseSource'\> | Create a dask bag of one data document for each row of the source table                                                    |
| <class 'list'\>, <class 'pathlib.Path'\>       | create a bag with one document for each file that was foundFrom this point we can hand off the logic to str(Bag) pipeline. |

*name*
: `<DocumentBag>.x('docs') or <DocumentBag>.docs`

*return type*
: <class 'dask.bag.core.Bag'\> | typing.Any

*supports pipeline flows*
: <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### e

Alias for: 

\* exploded-\>e (output)

*name*
: `<DocumentBag>.x('e') or <DocumentBag>.e`

*return type*
: typing.Callable[..., ForwardRef('DocumentBag')]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### exploded



*name*
: `<DocumentBag>.x('exploded') or <DocumentBag>.exploded`

*return type*
: typing.Callable[..., ForwardRef('DocumentBag')]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### file_path_list

| pipe_type              | description                                                                       |
|:-----------------------|:----------------------------------------------------------------------------------|
| <class 'list'\>         | Applies any function on items in a dask bag and filters them based on the result. |
|                        |     if func returns False, the element will be dropped from the bag.              |
| <class 'pathlib.Path'\> | No documentation                                                                  |

*name*
: `<DocumentBag>.x('file_path_list') or <DocumentBag>.file_path_list`

*return type*
: <class 'dask.bag.core.Bag'\> | typing.Any

*supports pipeline flows*
: <class 'list'\>, <class 'pathlib.Path'\>

### get_dicts

Returns a function closure which returns a bag of the specified
    property of the enclosed documents.

*name*
: `<DocumentBag>.x('get_dicts') or <DocumentBag>.get_dicts`

*return type*
: typing.Callable[[typing.Any], dask.bag.core.Bag]

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### index_column

No documentation

*name*
: `<DocumentBag>.x('index_column') or <DocumentBag>.index_column`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'pydoxtools.document.DatabaseSource'\>

### paths



*name*
: `<DocumentBag>.x('paths') or <DocumentBag>.paths`

*return type*
: typing.Callable

*supports pipeline flows*
: <class 'list'\>, <class 'pathlib.Path'\>

### root_path

Alias for: 

\* source-\>root_path (output)

*name*
: `<DocumentBag>.x('root_path') or <DocumentBag>.root_path`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'list'\>, <class 'pathlib.Path'\>

### sql

No documentation

*name*
: `<DocumentBag>.x('sql') or <DocumentBag>.sql`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'pydoxtools.document.DatabaseSource'\>

### stats

gather a number of statistics from documents as a pandas dataframe

*name*
: `<DocumentBag>.x('stats') or <DocumentBag>.stats`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### take

No documentation

*name*
: `<DocumentBag>.x('take') or <DocumentBag>.take`

*return type*
: typing.Any

*supports pipeline flows*
: <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### vectorizer

vectorizes a query, using the document configuration of the Documentbag to determine which model to use.

*name*
: `<DocumentBag>.x('vectorizer') or <DocumentBag>.vectorizer`

*return type*
: typing.Any

*supports pipeline flows*
: \*, <class 'dask.bag.core.Bag'\>, <class 'list'\>, <class 'pathlib.Path'\>, <class 'pydoxtools.document.DatabaseSource'\>

### Configuration parameters

| name               | description                                                                                                                                                    | default_values   |
|:-------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| bytes_per_chunk    | Configuration for values:                                                                                                                                      | 256 MiB          |
|                    |                                                                                                                                                                |                  |
|                    | \* bytes_per_chunk = 256 MiB (default)                                                                                                                         |                  |
| doc_configuration  | We can pass through a configuration object to Documents that are created in our document bag. Any setting that is supported by Document can be specified here. | {}               |
| forgiving_extracts | When enabled, if we execute certain batch operations on our document bag, this will not stop the extraction, but rather put an error message in the document.  | False            |
| verbosity          | Configuration for values:                                                                                                                                      | None             |
|                    |                                                                                                                                                                |                  |
|                    | \* verbosity = None (default)                                                                                                                                  |                  |