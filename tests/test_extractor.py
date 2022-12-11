import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

import io
import pathlib
from pathlib import Path

from pydoxtools import Document

logger = logging.getLogger(__name__)

test_files = [
    "./data/PFR-PR23_BAT-110__V1.00_.pdf",
    "./data/Datasheet-Centaur-Charger-DE.6f.pdf",
    # TODO: integrate OCR
    #"./data/north_american_countries.png",
    #"./data/north_american_countries.tif", "./data/north_american_countries.jpg",
    "./data/List of North American countries by population - Wikipedia.pdf",
    "./data/berrybase_raspberrypi4.html",
    "./data/test.html",
    "./data/remo-m_fixed-wing.2f.pdf",
    # TODO: file:///home/tom/git/pydoxtools/tests/data/alan_turing.txt
]

test_dir_path = pathlib.Path(__file__).parent.absolute()


def make_path_absolute(f: Path | str):
    return test_dir_path / Path(f)


# make directories work in pytest
test_files = [make_path_absolute(f) for f in test_files]


def run_single_non_interactive_document_test(file_name):
    logger.info(f"testing: {file_name}")
    # load object from path
    doc = Document(fobj=pathlib.Path(file_name))
    doc.run_all_extractors()
    assert doc._cache_hits >= 0
    doc_type = doc.document_type

    with open(file_name, "rb") as file:
        doc_str = file.read()

    # from bytestream
    doc = Document(fobj=io.BytesIO(doc_str), document_type=doc_type)
    doc.document_type
    doc.run_all_extractors()
    assert doc._cache_hits >= 0

    # from bytes
    doc = Document(fobj=doc_str, document_type=doc_type)
    doc.document_type
    doc.run_all_extractors()
    assert doc._cache_hits >= 0

    return doc


def test_string_extraction():
    with open(make_path_absolute("./data/alan_turing.txt"), "r") as f:
        some_string = f.read()

    doc = Document(fobj=some_string)
    doc.document_type
    doc.run_all_extractors()
    assert doc._cache_hits >= 0
    assert doc.keywords == {"Turing"}


def test_installation():
    from pydoxtools import nlp_utils
    nlp_utils.download_spacy_nlp_models(["md"])  # "also ["lg","trf"] etc.."


def test_all_documents():
    for f in test_files:
        logger.info(f"testing with {f}")
        doc = run_single_non_interactive_document_test(f)


def test_table_extraction():
    doc = Document(fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"))
    metrics = [t.metrics_X for t in doc.x("table_candidates") if t.is_valid]
    assert len(metrics) == 2
    assert doc.x("tables_df")[0].shape == (10, 2)
    assert doc.x("tables_df")[1].shape == (14, 2)


def test_qam_machine():
    doc = Document(
        fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"),
        config=dict(trf_model_id='distilbert-base-cased-distilled-squad')
    )
    answers = doc.x('answers')(questions=('what is this the product name?', 'who build the product?'))
    assert answers[0][0][0] == 'BST BAT - 110'
    assert answers[1][0][0] == 'The BST BAT - 110'


def test_multiple_product_extraction():
    pass


if __name__ == "__main__":
    # test if we can actually open the pdf...
    # with open("ocrpdf", "wb") as f:
    #    f.write(doc.ocr_pdf_file)
    # doc = Document(fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"))
    doc = Document(fobj=make_path_absolute("./data/Datasheet-Centaur-Charger-DE.6f.pdf"))
    # doc = Document(fobj=make_path_absolute("./data/north_american_countries.png"))
    # doc = Document(fobj=make_path_absolute("./data/berrybase_raspberrypi4.html"))
    # doc = Document(fobj=make_path_absolute("./data/remo-m_fixed-wing.2f.pdf"))
    # doc = Document(fobj=make_path_absolute("./data/north_american_countries.tif"))
    with open(make_path_absolute("./data/alan_turing.txt"), "r") as f:
        some_string = f.read()

    doc = Document(fobj=some_string)
    doc.document_type
    doc.run_all_extractors()

    # doc.run_all_extractors()

    # doc
    # doc.run_all_extractors()
    # index = doc.noun_index
    # keywords = doc.keywords

    # test_single_product_extraction()
    # test_qam_machine()

    # doc.run_all_extractors()
    # run_single_non_interactive_document_test("./data/berrybase_raspberrypi4.html")
    # test_table_extraction()
    # doc.x('answers', questions=('what is this the product name?', 'who build the product?'))
    pass
