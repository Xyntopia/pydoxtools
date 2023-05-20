#!/bin/sh

# copy README.md into docs for easy access
cp README.md docs/readme_cp.md
cp LICENSE DEVELOPMENT.md docs/

# build images
cd tests
python test_documentation_generation.py
cd ..

# Build the MkDocs project and deploy to github
mkdocs gh-deploy -r github