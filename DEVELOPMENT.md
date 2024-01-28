# Development & Contribution

The graph model of the library makes it very easy to extend it with new functionality.

- the document can be used as a base-model and overwritten with changes
- the graph can be changed dynamically
- new functions can be very easily integrated

## Installation from other branches

In order to install pydoxtools from a development branch "development_branch" you can do this:

pip install -U "pydoxtools[etl,inference] @ git+https://github.com/xyntopia/pydoxtools.git@development_branch"

## Pydoxtools Architecture

--> refer to "document"

## Testing

For unit-testing, the test dataset is needed. You can download it with this commmand:

```bash
poetry run clone-data
```

As of Jan 2024, only one dataset with some proprietary data is available an "open" dataset is being worked on. Contact
the author of pydoxtools if you would like to have access to the test dataset.


## Contribution Guidelines