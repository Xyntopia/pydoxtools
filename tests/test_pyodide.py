import pytest

from pydoxtools.document import Document

html_documents = [
    "./data/berrybase_raspberrypi4.html",
    "./data/test.html"
]


def test_html_text_extraction(html_document):
    """Test extracting text from an HTML document."""
    # Assuming Document can handle HTML files and extract text correctly
    doc = Document(fobj=str(html_documents[0]))
    assert "Heading" in doc.full_text
    assert "This is a test paragraph." in doc.full_text


if __name__ == "__main__":
    pytest.main()
