#!/bin/sh

# copy README.md into docs for easy access
cp README.md docs/README.md

# build images
cd tests
python -c "from test_extractor import test_logic_graph; test_logic_graph()"
cd ..

# Build the MkDocs project
mkdocs build