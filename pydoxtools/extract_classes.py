import langdetect
import pandas as pd

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
    def text_block_classes(self) -> pd.DataFrame:
        model: classifier.txt_block_classifier = classifier.load_classifier('text_block')
        blocks = self.textboxes
        classes = model.predict_proba(blocks).numpy()
        txtblocks = pd.DataFrame(self.textboxes, columns=["txt"])
        txtblocks[["add_prob", "ukn_prob"]] = classes.round(3)
        return txtblocks

    def addresses(self) -> list[str]:
        return self.text_block_classes[self.text_block_classes['add_prob'] > 0.5].txt.tolist()
