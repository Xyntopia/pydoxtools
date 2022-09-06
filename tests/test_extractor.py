import logging

from pydoxtools import Document

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

pdf_file_name = "../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf"
#with open(pdf_file_name) as pdf_file:
#    pdfdoc = Document(fobj=pdf_file)
#    pdfdoc.x('elements')

pdfdoc = Document(fobj=pdf_file_name)
pdfdoc.x('elements')



with open("../training_data/pdfs/product_page/berrybase_raspberrypi4.html") as html_file:
    htmldoc = Document(fobj=pdf_file)

logger.info("finished!")