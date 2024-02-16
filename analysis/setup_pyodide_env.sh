# create the environment
pyodide venv .venv-pyodide

# activate it
source .venv-pyodide/bin/activate

pip install poetry

# install pydoxtools for pyodide from dev dir

pip install -e .[pyodide]

# repositories:
https://github.com/goose3/goose3
https://github.com/WojciechMula/pyahocorasick

https://github.com/Mimino666/langdetect
https://github.com/buriy/python-readability
https://github.com/scrapinghub/extruct

