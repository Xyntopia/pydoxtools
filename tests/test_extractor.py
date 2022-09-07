import logging
from io import BytesIO

import pandas as pd
import pdfminer.jbig2

from pydoxtools import Document

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


#def test_file_loading():
if True:
    pdf_file_name = "../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf"
    with open(pdf_file_name, "rb") as pdf_file:
        pdfstr = pdf_file.read()

    pdfdoc = Document(fobj=BytesIO(pdfstr), document_type=".pdf")
    pdfdoc.document_type
    pdfdoc.x('elements')
    #pd.DataFrame([o.__dict__ for o in pdfdoc.x('elements')])

    pdfdoc = Document(fobj=pdf_file_name)
    pdfdoc.x('elements')

    with open("../training_data/pdfs/product_page/berrybase_raspberrypi4.html") as html_file:
        html_str = html_file.read()
    htmldoc = Document(fobj=html_str, document_type='.html')

    htmldoc.x('raw_txt')
    htmldoc.document_type

    logger.info("finished!")

#def check access to all standard extractors
if True:
    pdfdoc = Document(fobj=pdf_file_name)
    pdfdoc.x('elements')
    pdfdoc.document_type
    #print(pdfdoc.elements)
    for x in pdfdoc.x_funcs:
        pdfdoc.x(x)
    assert pdfdoc._cache_hits>0
    logger.info("finished")


if __name__=="__main__":
    #test_file_loading()
    pass