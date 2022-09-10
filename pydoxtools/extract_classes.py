import langdetect
import pandas as pd

from pydoxtools import classifier
from pydoxtools.document import Extractor


class LanguageExtractor(Extractor):
    def __call__(self, text) -> str:
        text = text.strip()
        if text:
            lang = langdetect.detect(text)
        else:
            lang = "unknown"
        return lang


class TextBlockClassifier(Extractor):
    def __init__(self, min_prob=0.5):
        super().__init__()
        self._min_prob = min_prob

    def __call__(self, text_box_elements: pd.DataFrame):
        model: classifier.txt_block_classifier = classifier.load_classifier('text_block')
        classes = model.predict_proba(text_box_elements.text).numpy()
        txtblocks = text_box_elements[["text"]].copy()
        txtblocks[["add_prob", "ukn_prob"]] = classes.round(3)
        return txtblocks[txtblocks['add_prob'] > self._min_prob].text.tolist()
