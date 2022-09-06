import logging
from io import BytesIO

from pydoxtools import Document

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

pdf_file_name = "../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf"
with open(pdf_file_name, "rb") as pdf_file:
    pdfstr = pdf_file.read()

pdfdoc = Document(fobj=BytesIO(pdfstr))
pdfdoc.x('elements')

pdfdoc = Document(fobj=pdf_file_name)
pdfdoc.x('elements')

with open("../training_data/pdfs/product_page/berrybase_raspberrypi4.html") as html_file:
    html_str = html_file.read()
htmldoc = Document(fobj=html_str)

htmldoc.x('raw_txt')

logger.info("finished!")
