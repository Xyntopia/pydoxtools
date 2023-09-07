"""
Created on Mon Dec 16 12:07:52 2019

@author: Thomas Meschede
"""

# alternatives to camelot:
# https://github.com/tabulapdf/tabula for "stream-tables"
# 


from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import io
import logging
import math
import typing
from io import StringIO
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
import pdf2image
# TODO: evaluate tabula as an additional table-read mechanism
import pdfminer
import pdfminer.high_level
import pdfminer.pdfdocument
import pdfminer.psparser
import pikepdf
from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams
from pdfminer.layout import LTChar, LTCurve, LTFigure, LTTextLine, LTAnno
from pdfminer.layout import LTTextContainer
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import resolve1
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import pydoxtools.operators_base
from pydoxtools import document_base, list_utils

logger = logging.getLogger(__name__)

try:
    from functools import cached_property
except ImportError:
    from pydoxtools.class_utils import cached_property

# make slicing a bit more natural with multipl coordinates
idx = pd.IndexSlice

"""
# methods to  convert pdf to html/text:
  
> pdftohtml PFR-PR05-HRVI-6HD-Flyer-V1.00-SV003.pdf test.html

# the next one preserves layout in textfile
> pdftotext -layout PFR-PR05-HRVI-6HD-Flyer-V1.00-SV003.pdf output.txt
"""


def _set_log_levels():
    """default loglevels of the libraries used here are very verbose...
    so we can optionally decrease the verbosity here"""
    # logging.getLogger('pdfminer.pdfinterp').setLevel(logging.WARNING)
    # logging.getLogger('pdfminer.pdfdocument').setLevel(logging.WARNING)
    # logging.getLogger('pdfminer').setLevel(logging.WARNING)
    pdflogs = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('pdfminer')]
    for ll in pdflogs:
        ll.setLevel(logging.WARNING)
    # logging.getLogger('camelot').setLevel(logging.WARNING) #not needed anymore...


_set_log_levels()


def repair_pdf(pdf_file_path: str) -> str:
    """
    repairs pdf and saves it using new filename.
    pikepdf needs to be installed in oder for this to work.

    TODO: do this "in-memory" because our algorithm
            can now work in memory-only with pdfminer.six
    """
    # create a temporary file in memory:
    # prefer a ramfs for speed
    newfilepath = "/run/user/1000/outtmp.pdf"
    if not Path(newfilepath).is_dir():
        # if it doesn#t exist (for example in a container) use normal tmp directory...
        newfilepath = "/tmp/outtmp.pdf"
    # open using pikepdf and save pdf again to mitigate a lot
    # of the problems with pdfminer.six
    with pikepdf.open(pdf_file_path) as pdf:
        # num_pages = len(pdf.pages)
        # del pdf.pages[-1]
        # pdf.save(str(pdf_file)[-3:]+"bk.pdf")
        pdf.save(newfilepath)

    return newfilepath


class PDFRepairError(Exception):
    pass


def repair_pdf_if_damaged(function):
    """
    repairs a pdf if certain exceptions are thrown due to faulty
    pdf files.

    TODO: better description of the following
    The function tobe-wrapped HAS to have "pdf_file" as first parameter
    """

    @functools.wraps(function)
    def wrapper(pdf_file, *args, **kwargs):
        try:
            return function(pdf_file, *args, **kwargs)
        except pdfminer.psparser.PSSyntaxError:
            logger.debug("repairing pdf file using pikepdf")
            f_repaired = repair_pdf(pdf_file)
            return function(f_repaired, *args, **kwargs)
        except pdfminer.pdfdocument.PDFEncryptionError as E:
            logger.info(f"{pdf_file} might be encrypted, trying if it has an empty password... and repairing")
            # logger.exception(f"could not open pdf document {pdf_file} it might be encrypted... "
            #                 f"(sometimes with an empty password)?:\n ({E})")
            f_repaired = repair_pdf(pdf_file)
            # TODO: catch encryption errior from the above command and return a string
            # that says "pdf is encrypted" or something similar...
            # or raise another exception that says: pdf is definitly encrypted1
            return function(f_repaired, *args, **kwargs)
            # logger.info("try to repair pdf:")
            # repair_pdf(repair_pdf_if_damaged())
        except IndexError:
            logger.exception("some sort of index error might be caused by this bug here:  "
                             "https://github.com/pdfminer/pdfminer.six/issues/218")
            logger.debug("trying to repair pdf file using pikepdf")
            f_repaired = repair_pdf(pdf_file)
            return function(f_repaired, *args, **kwargs)
        except:
            logger.exception(f"Not able to process pdf: {pdf_file}")
            raise PDFRepairError(f"Not able to process pdf: {pdf_file}")

    return wrapper


class PDFImageRenderer(pydoxtools.operators_base.Operator):
    """Take a document PDF and render an image from it."""

    def __call__(self, fobj: bytes, dpi: int, page_numbers: list[int]) -> dict[str, dict[PIL.Image]]:
        images = {}
        for page in page_numbers:
            image = pdf2image.convert_from_bytes(
                fobj, dpi=dpi, first_page=page + 1, last_page=page + 1, use_cropbox=False
            )
            images[page] = image[0]
        return dict(images=images)


def meta_infos(f: io.IOBase):
    parser = PDFParser(f)

    doc = pdfminer.pdfdocument.PDFDocument(parser)

    try:
        pagenum = resolve1(doc.catalog['Pages'])['Count']
    except (AttributeError, TypeError):
        logger.warning(f"could not read pagenumber of {f}, trying the 'slow' method")
        pagenum = sum(1 for p in extract_pages(f))

    res = list_utils.deep_str_convert({
        **(doc.info[0]),
        "pagenum": pagenum
    })
    return res


class PDFFileLoader(pydoxtools.operators_base.Operator):
    """
    Loads a pdf file and can extract all kinds of information from it.

    - we extract all textlines with some metadata such as color, textsize
    - use that information to extract "outliers" which are assumed to hold
      important information and represent for example titles of textboxes
    - extract assumed titles by text-size and word count
    - extract lines which have "list" characters such as "*" or "-"
    - join lines that are part of the same "box" as "textboxes"
    - extract table data

    TODO: move extract_elements into the page class...
    """

    def __init__(
            self,
            # laparams=LAParams(detect_vertical=True, boxes_flow=-1.0, all_texts=False),
            laparams=LAParams(detect_vertical=False),
            **kwargs
    ):
        """
        :param laparams: An LAParams object from pdfminer.layout. If None, uses
        some default settings that often work well.

        LAParams(
           line_overlap=0.5,  # 0.5, are chars in the same line?
           char_margin=2.0,  # 2.0, max distance between chars in words
           word_margin=0.1,  # 0.1, max distance between words in line
           line_margin=0.5,  # 0.5, max distance between lines in box
           boxes_flow=+0.5,  # 0.5, box order
           detect_vertical=True,
           all_texts=False
        )
        """
        super().__init__()
        self._laparams = laparams
        self.docs(
            pages_bbox="Return the 'mediabox' property of the pdf page which gives"
                       " the size of the page of a pdf in 72 dpi,"
                       " which is the standard unit of measurement in pdfs.",
            meta="Extract metadata from pdf",
            pages="A list of the extracted pages of the pdf",
            elements="Extract a list of textelements from pdf: Textlines, Graphics, Figures")
        self.t(
            meta=typing.Any,
            elements=pd.DataFrame,
            pages=set[int],
            pages_bbox=np.ndarray
        )

    def __call__(self, fobj: bytes, page_numbers=None, max_pages=0):
        doc_obj = io.BytesIO(fobj)
        docelements, extracted_page_numbers, pages_bbox = self.extract_pdf_elements_pdfsix_version(
            doc_obj, page_numbers, max_pages)
        meta = meta_infos(doc_obj)

        return dict(
            meta=meta,
            elements=docelements,
            pages=extracted_page_numbers,
            pages_bbox=pages_bbox
        )

    def extract_pdf_elements(self, fobj, page_numbers, max_pages):
        """
        extract pdf elements "manually"  which should be a bit faster due
        to better textbox algorithms
        """

        raise NotImplementedError("This needs some work...")

        # read the docs here:   https://euske.github.io/pdfminer/programming.html#overview
        output_string = StringIO()
        with open('samples/simple1.pdf', 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager(caching=True)
            #    device = PDFPageAggregator(resource_manager, laparams=laparams)
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())

            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        print(output_string.getvalue())

        """
        with open_filename(pdf_file, "rb") as fp:
            fp = cast(BinaryIO, fp)  # we opened in binary mode
            resource_manager = PDFResourceManager(caching=caching)
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)
            for page in PDFPage.get_pages(
                    fp, page_numbers, maxpages=maxpages, password=password, caching=caching
            ):
                interpreter.process_page(page)
                layout = device.get_result()
                yield layout
        """

        pass

    def extract_pdf_elements_pdfsix_version(self, fobj, page_numbers, max_pages):
        """
        extracts all text lines from a pdf and annotates them with various features.

        This is the "out-of-the-box" version of pdfminer.six it is sometimes very slow.
        Specifically for PDFs with a lot of graphic elements and small, distributed textboxes.
        TODO: make use of other pdf-pobjects as well (images, figures, drawings  etc...)
        TODO: check for already extracted pages and only extract missing ones...
        TODO: implement our own algorithm in order to identify textboxes...  the pdfminer.six
              one has problems with boxes when there is a line with a right- and a left justified
              text in the same line..  in most cases they should be split into two boxes...
        """
        docelements = []  # should be of type list[dict]
        # TODO: automatically classify text pieces already at this point here for example
        #       to find addresses, hint to tables etc... the rest of the algorithm would get a lot
        #       more precise this way...
        extracted_page_numbers = set()
        pages_bbox = {}

        # iterate through pages
        for page_layout in extract_pages(fobj,
                                         laparams=self._laparams,
                                         page_numbers=page_numbers,
                                         maxpages=max_pages):
            if page_numbers:
                page_num = page_numbers[page_layout.pageid - 1]
            else:
                page_num = page_layout.pageid - 1
            extracted_page_numbers.add(page_num)
            pages_bbox[page_num] = np.array(page_layout.bbox)
            # len(page_layout)
            # iterate through all page elements and translate them
            # TODO: make sure we adhere to a common schema for all file types here...
            for boxnum, element in enumerate(page_layout):
                docelement = None
                if isinstance(element, LTCurve):  # LTCurve are rectangles AND lines
                    # docelements should be compatible with document_base.DocumentElement
                    docelement = document_base.DocumentElement(
                        type=document_base.ElementType.Graphic,
                        obj=element,
                        linewidth=element.linewidth,
                        non_stroking_color=element.non_stroking_color,
                        stroking_color=element.stroking_color,
                        stroke=element.stroke,
                        fill=element.fill,
                        evenodd=element.evenodd,
                        p_num=page_num,
                        boxnum=boxnum,
                        x0=element.x0,
                        y0=element.y0,
                        x1=element.x1,
                        y1=element.y1
                    )
                    docelements.append(docelement)
                elif isinstance(element, LTTextContainer):
                    if isinstance(element, LTTextLine):
                        element = [element]
                    # linetext = ""
                    for linenum, text_line in enumerate(element):
                        fontset = set()
                        char_orientations = []
                        # TODO: this could be moved somewhere else and probably be made more efficient
                        for character in text_line:
                            if isinstance(character, LTChar):
                                charfont = document_base.Font(
                                    character.fontname, character.size,
                                    str(character.graphicstate.ncolor))
                                fontset.add(charfont)
                                rotation = math.atan2(character.matrix[1], character.matrix[0])
                                char_orientations.append(math.degrees(rotation))
                                mean_char_orientation = np.mean(char_orientations)
                            elif isinstance(character, LTAnno):
                                continue
                            else:  # couldbe LTAnno,
                                pass
                        linetext = text_line.get_text().strip()
                        if linetext.strip() == "":
                            continue
                        # extract metadata
                        # TODO: move most of these function to a "feature-generation-function"
                        # which extracts the information directly from the LTTextLine object
                        docelement = document_base.DocumentElement(
                            type=document_base.ElementType.Text,
                            char_orientations=char_orientations,
                            mean_char_orientation=mean_char_orientation,
                            obj=text_line,
                            rawtext=linetext,
                            font_infos=fontset,
                            p_num=page_num,
                            linenum=linenum,
                            boxnum=boxnum,
                            x0=text_line.x0,
                            y0=text_line.y0,
                            x1=text_line.x1,
                            y1=text_line.y1
                        )
                        docelements.append(docelement)
                elif isinstance(element, LTFigure):
                    docelement = document_base.DocumentElement(
                        type=document_base.ElementType.Image,
                        obj=element,
                        p_num=page_num,
                        boxnum=boxnum,
                        x0=element.x0,
                        y0=element.y0,
                        x1=element.x1,
                        y1=element.y1
                    )
                    docelements.append(docelement)
                else:
                    continue

        return docelements, extracted_page_numbers, pages_bbox
