import io
import logging
import pathlib

from pydoxtools import Document

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def run_single_document_test(file_name):
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
    # pd.DataFrame([o.__dict__ for o in pdfdoc.x('elements')])

    # TODO: from string

    logger.info(f"testing {file_name}")

    return doc


# def test_file_loading():
if True:
    test_files = [
        "../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf",
        "../training_data/pdfs/product_page/berrybase_raspberrypi4.html",
        #"../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf"
    ]

    for f in test_files:
        doc = run_single_document_test(f)

    doc = Document(fobj=pathlib.Path(test_files[1]))

if __name__ == "__main__":
    # test_file_loading()
    pass
