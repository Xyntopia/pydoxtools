import io
import logging
import pathlib
import pickle
from pathlib import Path

import pytest

from pydoxtools import settings
from pydoxtools.document import Document, DocumentSet
from pydoxtools.document_base import OperatorException
from pydoxtools.list_utils import flatten, iterablefyer

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

test_files_w_type = {
    "application/pdf": ["./data/PFR-PR23_BAT-110__V1.00_.pdf",
                        "./data/Datasheet-Centaur-Charger-DE.6f.pdf",
                        "./data/remo-m_fixed-wing.2f.pdf",
                        "./data/Doxcavator.pdf",
                        "./data/List of North American countries by population - Wikipedia.pdf"],
    "text/html": ["./data/berrybase_raspberrypi4.html",
                  "./data/test.html"],
    "text/plain": ["./data/alan_turing.txt"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "./data/demo.docx",
    # TODO: enable ODP
    # "./data/Doxcavator.odp",
    # TODO: enable pptx
    # "./data/Doxcavator.pptx",
    "application/vnd.oasis.opendocument.text": "./data/test.odt",
    "text/rtf": "./data/sample.rtf",
    "application/epub+zip": "./data/basic-v3plus2.epub",
    # images
    "image/png": "./data/north_american_countries.png",
    "image/tiff": "./data/north_american_countries.tif",
    "image/jpeg": "./data/north_american_countries.jpg",
    "text/markdown": "../README.md"
}
test_files = flatten(test_files_w_type.values())
test_dir_path = pathlib.Path(__file__).parent.absolute()


def make_path_absolute(f: Path | str):
    return test_dir_path / Path(f)


# make directories work in pytest
test_files = [make_path_absolute(f) for f in test_files]


def test_document_type_detection():
    for t,v in test_files_w_type.items():
        for f in iterablefyer(v):
            d = Document(f)
            assert d.document_type==t


def run_single_non_interactive_document_test(file_name):
    logger.info(f"testing: {file_name}")
    # load object from path
    doc = Document(fobj=pathlib.Path(file_name))
    doc.run_pipeline()
    assert doc._cache_hits >= 0
    doc_type = doc.document_type

    with open(file_name, "rb") as file:
        doc_str = file.read()

    # TODO: automatically recognize doc_type
    # from bytestream
    doc = Document(fobj=io.BytesIO(doc_str), document_type=doc_type)
    doc.document_type
    doc.run_pipeline()
    assert doc._cache_hits >= 0

    # from bytes
    doc = Document(fobj=doc_str, document_type=doc_type)
    doc.document_type
    doc.run_pipeline()
    assert doc._cache_hits >= 0

    return doc


def test_string_extraction():
    with open(make_path_absolute("./data/alan_turing.txt"), "r") as f:
        some_string = f.read()

    doc = Document(fobj=some_string)
    doc.document_type
    doc.run_pipeline()
    assert doc._cache_hits >= 30
    assert doc.keywords == ["Turing"]


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

    with open(make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"), "rb") as file:
        doc_str = file.read()
    doc = Document(fobj=doc_str, document_type=".pdf")
    metrics = [t.metrics_X for t in doc.x("table_candidates") if t.is_valid]
    assert len(metrics) == 2
    assert doc.x("tables_df")[0].shape == (10, 2)
    assert doc.x("tables_df")[1].shape == (14, 2)

    # TODO: add test to automatically recognize correct document


# TODO: test configuration of our class with dedicated Configuration Operators

# TODO: test property_dict, yaml & json output

# TODO: add questions on tables using chatgpt

def test_qam_machine():
    doc = Document(
        fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf")
    ).config(qam_model_id='deepset/minilm-uncased-squad2')
    answers = doc.answers(questions=(
        'what is this the product name?',
        'who build the product?',
        'what is the address'
    ))
    assert answers[0][0][0] == 'bst bat - 110'
    assert answers[1][0][0] == 'bst bat - 110'
    assert answers[2][0][0] == 'the bst'


def test_address_extraction():
    doc = Document(
        fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf")
    )
    addresses = doc.addresses
    findthis = """FURTHER INFORMATION   Max-Planck-Str. 3 | 12489 Berlin Germany | info@berlin-space-tech.com | www.berlin-space-tech.com"""
    assert findthis in addresses


def test_chat_gpt():
    import openai
    doc = Document(
        fobj=make_path_absolute("./data/sample.rtf")
    )
    try:
        ans = doc.chat_answers(["what is the text about?"])
    except openai.error.AuthenticationError:
        logger.info("openai chat test omitted, due to lack of API key.")


def test_pandoc():
    pandoc_files = [
        # TODO: add pandoc functionality to ordinary text files
        # "./data/alan_turing.txt",
        "./data/demo.docx",
        "./data/test.odt",
        "./data/sample.rtf",
        "./data/basic-v3plus2.epub"
    ]

    import pandoc.types
    for f in pandoc_files:
        logger.info(f"testing pandoc with {f}")
        doc = Document(fobj=make_path_absolute(f))
        assert isinstance(doc.pandoc_blocks, list)
        try:
            assert isinstance(doc.pandoc_blocks[0], pandoc.types.Block)
        except:
            logger.warning(f"no blocks in the file {f}!")

    doc = Document(fobj=make_path_absolute("./data/sample.rtf"))
    assert len(doc.tables_df) == 1
    assert len(doc.headers) == 2
    # TODO: this one doesn't recognize lists yet...

    doc = Document(fobj=make_path_absolute("./data/demo.md"), document_type=".markdown")
    assert len(doc.tables) == 5
    assert len(doc.lists) == 6

    doc = Document(fobj=make_path_absolute("./data/demo.docx"), document_type=".docx")
    assert len(doc.tables) == 5
    assert len(doc.lists) == 6


def test_pipeline_graph():
    doc = Document(fobj=make_path_absolute("./data/demo.docx"), document_type=".docx")
    # TODO: generate graphs for all document types
    for k in doc._operators:
        doc.pipeline_graph(
            image_path=settings._PYDOXTOOLS_DIR / f"docs/images/document_logic_{k}.svg",
            document_logic_id=k
        )


def test_documentation_generation():
    doc = Document
    docs = doc.pipeline_docs()

    pipeline_docs = f"""
# Pipelines

This documents the output values of the nodes of each pipeline that 
can be accessed through the pipeline interface.

Pipeline visualizations for every supported file type can be found
[here](https://github.com/Xyntopia/pydoxtools/tree/gh-pages/images).

## [pydoxtools.Document][]

{docs}
""".strip()
    with open(make_path_absolute('../docs/pipelines.md'), "w") as f:
        f.write(pipeline_docs)


def test_pipeline_configuration():
    with open(make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"), "rb") as file:
        doc_str = file.read()
    doc = Document(
        fobj=doc_str, document_type=".pdf"
    ).config(qam_model_id='deepset/roberta-base-squad2')
    ans = doc.x('answers')(questions=('what is the address?',))
    # doc.config(qam_model_id='some_model_that_doesn_t_exist')
    # ans = doc.x('answers')(questions=('what is the address?',))


def test_url_download():
    doc = Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf",
        document_type=".pdf"
    )
    # doc = Document(fobj=make_path_absolute("./data/alan_turing.txt"))
    # doc = run_single_non_interactive_document_test("./data/alan_turing.txt")
    assert doc.x("tables_df")[0].shape == (8, 3)


def test_ocr_and_exceptions():
    with pytest.raises(OperatorException) as exc_info:
        doc = Document(
            fobj=make_path_absolute("./data/north_american_countries.tif"),
            mime_type="image/tiff"
        ).config(ocr_on=False)
        doc.full_text
        assert exc_info.args == ('could not get ocr_pdf_file!',)

    doc = Document(
        fobj=make_path_absolute("./data/north_american_countries.tif"),
        mime_type="image/tiff"
    ).config(ocr_on=True)
    doc.full_text


# TODO: add tests for dict & yaml
def test_yaml_json_dict_prop_dict():
    # document chaining
    doc = Document(fobj=make_path_absolute("../README.md")).config(spacy_model_size='trf')
    doc.property_dict(
        "document_type",
        "num_words",
        "language",
        "entities",
        "keywords"
    )
    doc.textrank_sents
    doc.keywords
    doc.property_dict("addresses", "filename", "keywords")
    doc.yaml("addresses", "filename", "keywords")
    doc.json("addresses", "filename", "keywords")
    a = Document(doc.yaml(
        "document_type",
        "num_words",
        "language",
        "entities",
        "keywords"
    ), document_type=".yaml")
    d = Document(a.data, document_type="dict")


def test_summarization():
    doc = Document(fobj=make_path_absolute("../README.md")).config(spacy_model_size='trf')
    doc.keywords
    summary = doc.summary

    doc = Document(
        fobj=make_path_absolute("../README.md"),
        document_type=".md"
    ).config(spacy_model_size='md')
    doc.keywords
    doc.textrank_sents


def test_document_pickling():
    # Create a sample Document object
    # TODO: add some efficiency checks
    doc = Document(fobj=make_path_absolute("./data/alan_turing.txt"))
    doc.keywords

    # Pickle the Document object
    pickled_doc = pickle.dumps(doc)

    # Unpickle the Document object
    unpickled_doc = pickle.loads(pickled_doc)
    unpickled_doc.keywords

    # Assert that the original and unpickled Document objects are equal
    assert doc.keywords == unpickled_doc.keywords


def test_sql_download():
    # import dask
    # dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
    # dask.config.set(scheduler='processes') # # overwrite default with multiprocessing scheduler
    # dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    # from dask.distributed import Client
    # client = Client()
    # or
    # client = Client(processes=False)

    connection_string = "postgresql://tnkaiypekofebmaxuxwlysbu%40psql-mock-database-cloud:" \
                        "yvzhpmjuconioczuiglnfnoq@psql-mock-database-cloud.postgres.database.azure.com:5432" \
                        "/cars1682799488882fyfxytaajychjrer"

    docs = DocumentSet(source=dict(
        connection_string=connection_string,
        sql="users",
        index_column="id"
    ), pipeline="db", max_documents=1000)
    # d = docs.props_bag(["vector"]).take(3)
    vector_bag = docs.props_bag(["source", "text_segment_vectors"])
    vector_bag.push_sql(source=dict(
        connection_string=connection_string,
        sql="users",
        index_column="id"
    ))

def test_dict():
    person_document = {'full_name': 'Susan Williamson',
                       'first_name': 'Susan',
                       'last_name': 'Williamson',
                       'username': 'sWilliamson',
                       'email': 'Kyla_Considine@hotmail.com',
                       'email_verified': True,
                       'phone': '808.797.1741',
                       'twitter_handle': 'sWilliamson',
                       'bio': 'Friendly music geek. Organizer. Twitter scholar. Creator. General food nerd. Future teen idol. Thinker.'}
    doc = Document(person_document, document_type="dict")
    doc.keywords


if __name__ == "__main__":
    # test if we can actually open the pdf...
    # with open("ocrpdf", "wb") as f:
    #    f.write(doc.ocr_pdf_file)

    # test_qam_machine()
    test_document_type_detection()
    test_dict()
    test_document_pickling()
    test_documentation_generation()
    test_pipeline_graph()
    # test_sql_download()

    if False:
        with open(make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"), "rb") as file:
            doc_str = file.read()
        doc = Document(
            fobj=doc_str, document_type=".pdf"
        ).config(qam_model_id='deepset/roberta-base-squad2')
        ans = doc.x('answers')(questions=('what is the address?',))
        doc.addresses

    # TODO: load a "wrong" configuration and make sure, this spits out an error :)

    # graph_string = doc.logic_graph()

    # doc = Document(fobj=make_path_absolute("./data/north_american_countries.png"))
    # doc = Document(fobj=make_path_absolute("./data/berrybase_raspberrypi4.html"))
    # doc = Document(fobj=make_path_absolute("./data/remo-m_fixed-wing.2f.pdf"))
    # doc = Document(fobj=make_path_absolute("./data/north_american_countries.tif"))
    # with open(make_path_absolute("./data/alan_turing.txt"), "r") as f:
    #    some_string = f.read()
    #    doc = Document(fobj=some_string)

    # doc.run_all_operators()

    # doc.run_all_operators()

    # doc
    # doc.run_all_operators()
    # index = doc.noun_index
    # keywords = doc.keywords

    # test_single_product_extraction()
    # test_qam_machine()

    # doc.run_all_operators()
    # run_single_non_interactive_document_test("./data/berrybase_raspberrypi4.html")
    # test_table_extraction()
    # doc.x('answers', questions=('what is this the product name?', 'who build the product?'))
    pass
