from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import io
import logging

import langdetect
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text

from pydoxtools import ocr_language_mappings
from pydoxtools.operators_base import Operator

logger = logging.getLogger(__name__)


class OCRException(Exception):
    pass


class OCRExtractor(Operator):
    """
    Takes an image encoded in bytes and returns a pdf document
    which can be used to extract data.

    TODO: maybe we could add "lines" here and detect other thigns such as images,
          figures  etc...?
    """

    def __call__(self, file: bytes, ocr_on: bool = True, ocr_lang="auto"):
        if not ocr_on:
            raise OCRException("OCR is not enabled!!")
        file = Image.open(io.BytesIO(file))
        if ocr_lang == "auto":
            pdf = pytesseract.image_to_pdf_or_hocr(file, extension='pdf', lang=None)
            text = extract_text(io.BytesIO(pdf))
            try:
                lang = langdetect.detect(text)
            except langdetect.lang_detect_exception.LangDetectException as err:
                if err.args[0] == 'No features in text.':
                    lang = "en"  # simply use english as a language
                else:
                    raise OCRException("could not detect language !!!")
            # get the corresponding language for tesseract
            lang = ocr_language_mappings.langdetect2tesseract.get(lang, None)
            file.seek(0)  # we are scanning th same file now with the correct language
        else:
            lang = ocr_lang

        pdf = pytesseract.image_to_pdf_or_hocr(file, extension='pdf', lang=lang)

        return pdf
