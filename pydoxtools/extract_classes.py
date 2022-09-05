from document import Extractor
from functools import cached_property

class LanguageExtractor(Extractor):
    @cached_property
    def lang(self) -> str:
        text = self.full_text.strip()
        if text:
            lang = langdetect.detect(text)
        else:
            lang = "unknown"
        return lang

class TextBlockClassifier(Extractor):
    @cached_property
    def text_block_classes(self) -> pd.DataFrame:
        model: classifier.txt_block_classifier = classifier.load_classifier('text_block')
        blocks = self.textboxes
        classes = model.predict_proba(blocks).numpy()
        txtblocks = pd.DataFrame(self.textboxes, columns=["txt"])
        txtblocks[["add_prob", "ukn_prob"]] = classes.round(3)
        return txtblocks

    @cached_property
    def addresses(self) -> List[str]:
        return self.text_block_classes[self.text_block_classes['add_prob'] > 0.5].txt.tolist()
