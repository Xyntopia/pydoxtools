from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import io
import logging
import pathlib
import pickle
from pathlib import Path

import pytest

import pydoxtools
import pydoxtools.document
from pydoxtools.document import Document, DocumentBag
from pydoxtools.list_utils import flatten, iterablefyer
from pydoxtools.operators_base import OperatorException
from pydoxtools.settings import settings

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

test_files_w_type = {
    # TODO: enable ODP
    # "./data/Doxcavator.odp",
    "mediawiki": "./data/Starship.xml",
    # images
    "image/png": "./data/north_american_countries.png",
    "image/tiff": "./data/north_american_countries.tif",
    "image/jpeg": "./data/north_american_countries.jpg",
    "text/markdown": ["./data/demo.md",
                      "../README.md", ],
    "text/html": ["./data/berrybase_raspberrypi4.html",
                  "./data/test.html"],
    "text/rtf": "./data/sample.rtf",
    "application/pdf": ["./data/PFR-PR23_BAT-110__V1.00_.pdf",
                        "./data/Datasheet-Centaur-Charger-DE.6f.pdf",
                        "./data/remo-m_fixed-wing.2f.pdf",
                        "./data/Doxcavator.pdf",
                        "./data/List of North American countries by population - Wikipedia.pdf"],
    "text/plain": ["./data/alan_turing.txt"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "./data/demo.docx",
    # TODO: enable pptx
    # "./data/Doxcavator.pptx",
    "application/vnd.oasis.opendocument.text": "./data/test.odt",
    "application/epub+zip": "./data/basic-v3plus2.epub",
}
test_files = flatten(test_files_w_type.values())
test_dir_path = pathlib.Path(__file__).parent.absolute()


def make_path_absolute(f: Path | str):
    return test_dir_path / Path(f)


# make directories work in pytest
test_files = [make_path_absolute(f) for f in test_files]


def test_document_type_detection():
    for t, v in test_files_w_type.items():
        for f in iterablefyer(v):
            d = Document(make_path_absolute(f))
            logger.info(f"testing filetype detection for {f}")
            assert d.document_type == t

            # TODO: automatically recognize doc_type
            # from bytestream
            if d.magic_library_available():
                source = None
            else:
                source = f

            d = Document(str(make_path_absolute(f)))
            logger.info(f"testing filetype string path detection for {f}")
            assert d.document_type == t

            with open(make_path_absolute(f), "rb") as file:
                doc_str = file.read()
            d = Document(fobj=doc_str, source=source)
            assert d.document_type == t
            d = Document(fobj=io.BytesIO(doc_str), source=source)
            assert d.document_type == t

            with open(make_path_absolute(f), "rb") as file:
                d = Document(fobj=file)
                assert d.document_type == t

    d = Document('../README.md', document_type="string")
    assert d.document_type == "string"
    assert '../README.md' == d.full_text

    d = Document('../README.md')
    assert d.document_type == "text/markdown"
    assert '../README.md' not in d.full_text


# TODO: implement a test case where we need to overwrite the document
#       detection because the document is recognized in some funny way...

# TODO: test loading all kinds of forms of documents:
#       string, path-string, Path, byte,

# load rtf as string and file and path


def run_single_non_interactive_document_test(file_name):
    logger.info(f"testing: {file_name}")
    # load object from path
    doc = Document(fobj=pathlib.Path(file_name))
    doc.run_pipeline_fast()
    doctype = doc.document_type
    assert doc
    assert doc._stats["cache_hits"] >= 0

    with open(file_name, "rb") as file:
        doc_str = file.read()

    # TODO: automatically recognize doc_type
    # from bytestream
    if doc.magic_library_available():
        source = None
    else:
        source = file_name

    doc = Document(fobj=io.BytesIO(doc_str), source=source)
    assert doc.document_type == doctype
    doc.run_pipeline_fast()
    assert doc._stats["cache_hits"] >= 0

    # from bytes
    doc = Document(fobj=doc_str, source=source)
    assert doc.document_type == doctype
    doc.run_pipeline_fast()
    assert doc._stats["cache_hits"] >= 0

    return doc


def test_string_extraction():
    file = make_path_absolute("./data/alan_turing.txt")
    doc = Document(source=str(file))
    doc.document_type
    doc.keywords
    doc.run_pipeline_fast()

    with open(file, "r") as f:
        some_string = f.read()

    doc = Document(fobj=some_string)
    doc.document_type
    doc.run_pipeline_fast()
    assert doc._stats["cache_hits"] >= 30
    assert doc.keywords == ["Turing"]


def test_all_documents():
    for f in test_files:
        logger.info(f"testing with {f}")
        doc = run_single_non_interactive_document_test(f)


def test_table_extraction():
    fpath = make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf")
    doc = Document(fobj=fpath)
    metrics = [t.metrics_X for t in doc.x("table_candidates") if t.is_valid]
    assert len(metrics) == 2
    assert doc.x("tables_df")[0].shape == (10, 2)
    assert doc.x("tables_df")[1].shape == (14, 2)

    with open(fpath, "rb") as file:
        doc_str = file.read()
    doc = Document(fobj=doc_str, source=None if Document.magic_library_available() else fpath)
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
        fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"),
        configuration=dict(qam_model_id='deepset/minilm-uncased-squad2'))
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
        fobj=make_path_absolute("./data/sample.rtf"),
        chat_model_id='gpt-3.5-turbo'
    )
    try:
        ans = doc.chat_answers(["what is the text about?"])
    except openai.error.AuthenticationError:
        logger.info("openai chat test omitted, due to lack of API key.")


def test_gpt4all():
    doc = Document(
        fobj=make_path_absolute("./data/sample.rtf"),
        chat_model_id='ggml-mpt-7b-instruct'
    )
    ans = doc.chat_answers(["what is the text about?"])


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

    doc = Document(fobj=make_path_absolute("./data/demo.md"))
    assert len(doc.tables) == 5
    assert len(doc.lists) == 6

    doc = Document(fobj=make_path_absolute("./data/demo.docx"))
    assert len(doc.tables) == 5
    assert len(doc.lists) == 6


def test_pipeline_configuration():
    doc = Document(fobj=make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf"),
                   configuration=dict(
                       qam_model_id='deepset/roberta-base-squad2'))
    ans = doc.x('answers')(questions=('what is the address?',))


def test_url_download():
    doc = Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf",
        document_type="auto" if Document.magic_library_available() else 'application/pdf'
    )
    # doc = Document(fobj=make_path_absolute("./data/alan_turing.txt"))
    # doc = run_single_non_interactive_document_test("./data/alan_turing.txt")
    assert doc.x("tables_df")[0].shape == (8, 3)


def test_ocr_and_exceptions():
    doc = Document(fobj=make_path_absolute("./data/north_american_countries.tif"),
                   configuration=dict(ocr_on=True))
    doc.full_text

    with pytest.raises(OperatorException) as exc_info:
        doc = Document(fobj=make_path_absolute("./data/north_american_countries.tif"),
                       configuration=dict(ocr_on=False))
        doc.full_text
        assert exc_info.args == ('could not get ocr_pdf_file!',)


# TODO: add tests for dict & yaml
def test_yaml_json_dict_prop_dict():
    # document chaining
    doc = Document(fobj=make_path_absolute("../README.md"),
                   configuration=dict(spacy_model_size='trf'))
    doc.to_dict(
        "document_type",
        "num_words",
        "language",
        "entities",
        "keywords"
    )
    doc.textrank_sents
    doc.keywords
    doc.to_dict("addresses", "filename", "keywords")
    doc.to_yaml("addresses", "filename", "keywords")
    doc.to_json("addresses", "filename", "keywords")
    a = Document(doc.to_yaml(
        "document_type",
        "num_words",
        "language",
        "entities",
        "keywords"
    ), document_type="application/x-yaml")
    d = Document(a.data)


def test_summarization():
    doc = Document(fobj=make_path_absolute("../README.md"),
                   configuration=dict(spacy_model_size='trf'))
    doc.configuration
    doc.full_text
    doc.keywords
    summary = doc.slow_summary
    # check if certain keywords appear in the summary
    wn = [w for w in {"Pydoxtools", "library", "AI", "Pipeline", "table",
                      "pdf", "documents", "pipelines", "analysis"}
          if w.lower() in summary.lower()]
    assert len(wn) > 5

    doc = Document(fobj=make_path_absolute("../README.md"),
                   configuration=dict(spacy_model_size='md'))
    doc.keywords
    doc.textrank_sents


def test_document_pickling():
    # Create a sample Document object
    # TODO: add some efficiency checks
    doc = Document(fobj=make_path_absolute("./data/alan_turing.txt"))
    assert doc.keywords == ['Turing']  # ['mathematical biology','Victoria University','Turing','accidental poisoning']

    # Pickle the Document object
    pickled_doc = pickle.dumps(doc)

    # Unpickle the Document object
    unpickled_doc = pickle.loads(pickled_doc)

    # Assert that the original and unpickled Document objects are equal
    assert doc.keywords == unpickled_doc.keywords


def test_sql_download():
    import dask
    # dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
    # dask.config.set(scheduler='processes') # # overwrite default with multiprocessing scheduler
    # from dask.distributed import Client
    # client = Client()
    # or
    # client = Client(processes=False)

    dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging

    home = Path.home()
    database_source = pydoxtools.document.DatabaseSource(
        connection_string="sqlite:///" + str(home / "comcharax/data/component_pages.db"),
        sql="component_pages",
        index_column="id"
    )

    docs = DocumentBag(source=database_source, pipeline="db", max_documents=10)
    # d = docs.props_bag(["vector"]).take(3)
    df = docs.dataframe.get_partition(0)

    vector_bag = docs.get_dicts("data", "text_segment_vectors")
    data = vector_bag.take(1)
    new_bag = docs.get_dicts("source", "full_text")
    new_bag.take(1)
    # TODO: this doesn't work anymore, because we can not accept bags with
    #       arbitrary data anymore
    # text_bag = DocumentBag(new_bag)
    # text_bag.get_dicts("vector").take(1)
    d = docs.docs.take(1)[0]
    d.keys
    d.values
    d.text_segments
    d.data['index']

    ft = docs.apply(lambda d: d.data["raw_html"]).take(2)[0].full_text
    html_doc_bag = docs.apply(lambda d: [d.data["raw_html"], d.data["index"]])
    idx = html_doc_bag.get_dicts("source", "vector", "full_text")
    logger.info("converting sql table to dataframe")
    dd = idx.to_dataframe()


def test_dict():
    person_document = {'full_name': 'Susan Williamson',
                       'first_name': 'Susan',
                       'last_name': 'Williamson',
                       'username': 'sWilliamson',
                       'email': 'Kyla_Considine@hotmail.com',
                       'email_verified': True,
                       'phone': '808.797.1741',
                       'twitter_handle': 'sWilliamson',
                       'bio': 'Friendly music geek. Organizer. Twitter scholar. Creator. General food nerd. '
                              'Future teen idol. Thinker.'}
    doc = Document(person_document,
                   configuration=dict(summarizer_max_text_len=400))
    doc2 = Document(person_document)
    doc.run_pipeline_fast()
    assert doc.keywords == ['Susan full_name', 'Susan Williamson', 'bio', 'Organizer', 'Twitter scholar']
    assert doc.document_type == "<class 'dict'>"


def test_nlp_utils():
    from pydoxtools import nlp_utils as nu
    import time

    doc = Document(make_path_absolute("../README.md"),
                   configuration=dict(
                       vectorizer_only_tokenizer=False,
                       vectorizer_model="sentence-transformers/all-MiniLM-L6-v2"
                   ))
    defconf = doc.get_configuration_names("*")
    b = doc.configuration

    start_time = time.monotonic()
    emb = doc.embedding
    embs = doc.tok_embeddings
    tok = doc.tokens
    end_time = time.monotonic()
    # calculate the elapsed time
    print(f"Elapsed time: for embedding generation {end_time - start_time:.5f} seconds")

    tokenizer = nu.load_tokenizer(doc.vectorizer_model)
    ids = tokenizer.convert_tokens_to_ids(tok)
    txt = nu.convert_ids_to_string(doc.vectorizer_model, ids)

    doc.full_text
    doc.sents[5]
    nu.get_model_max_len(nu.load_model(doc.vectorizer_model))

    wemb, wtok = nu.fullword_embeddings(tok, embs)
    sd = doc.spacy_nlp(" ".join(wtok))


def test_disk_cache():
    d = Document(source=make_path_absolute("../README.md")).set_disk_cache_settings(
        enable=False,
        ttl=3600  # keep cache for 1 hour
    )
    kw = sorted(d.keywords)
    d = Document(source=make_path_absolute("../README.md")).set_disk_cache_settings(
        enable=True,
        ttl=3600  # keep cache for 1 hour
    )
    assert sorted(d.keywords) == kw
    d = Document(source=make_path_absolute("../README.md")).set_disk_cache_settings(
        enable=True,
        ttl=3600  # keep cache for 1 hour
    )
    assert sorted(d.keywords) == kw
    assert d._stats["cache_hits"] == 0
    assert d._stats["disk_cache_hits"] == 1

    settings.PDX_ENABLE_DISK_CACHE = True
    d = Document(source=make_path_absolute("../README.md"))
    d.keywords
    assert d._stats["cache_hits"] == 0
    assert d._stats["disk_cache_hits"] == 1
    settings.PDX_ENABLE_DISK_CACHE = False


def test_source_vs_fobj():
    """testing wether disk caching doesn't cause any collisions..."""
    settings.PDX_ENABLE_DISK_CACHE = True
    fobj = "../README.md"
    d = Document(fobj)
    d.full_text
    # ts = d.text_segments
    nd1 = Document("abcdef", source=d.source, document_type="string",
                   configuration=d.configuration)
    nd2 = Document("123456", source=d.source, document_type="string",
                   configuration=d.configuration)
    assert nd1.full_text == "abcdef"
    assert nd2.full_text == "123456"
    settings.PDX_ENABLE_DISK_CACHE = False


def test_old_bugs():
    d = Document(make_path_absolute("./data/Starship.xml"))
    d.document_type
    d.clean_text
    # assert d.document_type
    d.embedding


def test_erronous_file():
    d = Document(make_path_absolute('./data/random_not_openable_file.bin'))
    d.document_type
    d.text_segments
    d = Document(make_path_absolute('./data/Doxcavator.odp'))
    d.document_type
    d.text_segments
    d = Document(make_path_absolute('./data/erronous_pdf.pdf'))
    assert d.document_type != 'application/pdf'
    d.text_segments


def test_json_schema():
    test = Document.Model()


def test_typing():
    # TODO: actually test the types with some asserts...
    from pydoxtools.operators_base import FunctionOperator
    d = Document("asasd")
    docs = Document.operator_infos()
    optypes = Document.operator_types()
    m = Document.Model()()
    func = d._pipelines["*"]["full_text"]
    func = d._pipelines["*"]["a_d_ratio"]
    func = d._pipelines["*"]["embedding"]
    # func = d._pipelines["*"]["full_text"]
    # func = d._pipelines["*"]["chat_answers"]
    # t=func[str]("asdas")
    # a=func("asdasdas2123")
    func.return_type
    # typing.get_type_hints(func._func)
    # typing.get_type_hints(func.__call__)['return']
    f = FunctionOperator[float](lambda x: 1)
    Document.Model


def test_image_table_recognition():
    file = make_path_absolute("./data/PFR-PR23_BAT-110__V1.00_.pdf")
    pdf = Document(file)
    img = pdf.x("images")[0]

    d = Document(img)
    assert len(d.table_candidates) >= 1
    assert len(d.tables) == 1
    raise NotImplementedError("Correct text-based table readout!!")


def test_pdf_pages():
    doc = Document(make_path_absolute("./data/Doxcavator.pdf"))
    im1 = doc.images[8]
    assert doc.page_set == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

    doc = Document(make_path_absolute("./data/Doxcavator.pdf"), page_numbers=[2, 4])
    assert doc.page_set == {2, 4}

    doc = Document(make_path_absolute("./data/Doxcavator.pdf"), page_numbers=[8])
    assert "e.g. Searching for Insurance \nFraud\n" in doc.full_text

    new_image = doc.images[8]
    assert new_image == im1
    b = io.BytesIO()
    new_image.save(b, 'PNG')
    d = Document(b.getvalue())
    assert "e.g.  Searching  for  Insurance \n" in d.full_text


def test_pdf_text_extraction():
    training_data = pathlib.Path.home() / "comcharax/data"
    page = 15
    pdf_file = training_data / "sparepartsnow/06_Kraftspannfutter_Zylinder_Luenetten_2020.01_de_web.pdf"
    # %% jupyter={"outputs_hidden": true}

    # text = extract_text(pdf_file, page_numbers=[page])
    # print(text)

    pdf = Document(pdf_file, page_numbers=[page])
    pdf.full_text


def test_list_query():
    a = ['',
         'Id.-Nr.',
         'Futtergröße',
         'Backenlänge mm',
         'Backenhöhe mm',
         'Krallenlänge mm',
         'Zahnteilung']
    b = ['', '', '', '', '', '', '', '', '', '137039']
    doc = Document([a, b])
    doc.x("text_box_elements")
    doc.x("text_segments")
    doc.x("text_segment_ids")
    doc.x("text_segment_vecs")
    doc.x("text_segment_index")
    doc.vectorizer("test")
    doc.segment_query("['product' 'id' 'ID Nr' Nr. Id.]")


if __name__ == "__main__":
    # a = pd.DataFrame(sd.sents)
    # a[2]
    file = "/home/tom/git/doxcavator/backend/lib/componardo/pydoxtools/tests/data/PFR-PR23_BAT-110__V1.00_.pdf"
    # run_single_non_interactive_document_test(file)
    doc = Document(file)
    test_list_query()
    pass
