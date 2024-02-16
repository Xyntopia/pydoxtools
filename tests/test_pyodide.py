import pathlib
from pathlib import Path

import pytest

from pydoxtools.document import Document

test_dir_path = pathlib.Path(__file__).parent.absolute()


def make_path_absolute(f: Path | str):
    return test_dir_path / Path(f)


html_documents = [
    "./data/berrybase_raspberrypi4.html",
    "./data/test.html"
]


def test_html_text_extraction():
    """Test extracting text from an HTML document."""
    # Assuming Document can handle HTML files and extract text correctly
    doc = Document(fobj=make_path_absolute(html_documents[0]))
    print(doc.full_text)
    assert "Heading" in doc.full_text
    assert "This is a test paragraph." in doc.full_text


if __name__ == "__main__":
    pytest.main()
