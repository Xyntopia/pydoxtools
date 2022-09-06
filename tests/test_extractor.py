from pydoxtools.document import Document

with open("../training_data/pdfs/datasheet/BUSI-XCAM-SY-00011.22.pdf") as pdf_file:
    doc = Document(fobj = pdf_file)

