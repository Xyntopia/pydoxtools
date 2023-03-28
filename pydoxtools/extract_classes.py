import langdetect
import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

from pydoxtools.document_base import Extractor
from pydoxtools.settings import settings


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
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_name = "txtblockclassifier"
        model_dir = settings.MODEL_DIR / model_name
        # tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512, 'return_tensors': 'pt'}
        # TODO: optionally enable CUDA...
        # TODO: only extract "unique" addresses
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # .to("cuda")
        model = pipeline("text-classification", model=model, tokenizer=tokenizer)
        text = text_box_elements["text"].str.strip()
        res = text.apply(lambda x: model(
            [x], truncation=True, padding=True)[0]["label"])
        return text[res == "address"].to_list()
