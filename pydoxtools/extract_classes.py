from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import dataclasses
import functools
import logging
import typing

import langdetect
import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

import pydoxtools.document_base
from pydoxtools.operators_base import Operator
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)


class LanguageExtractor(Operator):
    def __call__(self, text) -> str:
        text = text.strip()
        if text:
            lang = langdetect.detect(text)
        else:
            lang = "unknown"
        return lang


def zero_shot_classifier(
        text: str,
        candidate_labels: list[str],
        zero_shot_model_name: str = "facebook/bart-large-mnli",
        multi_label: bool = True
) -> list[str]:
    """
    if model = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", we need to install
    sentencepiece like this:

        pip install sentencepiece
    """
    tokenizer = AutoTokenizer.from_pretrained(zero_shot_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(zero_shot_model_name)

    pipe = pipeline(task="zero-shot-classification", model=model, tokenizer=tokenizer)
    # pipe = pipeline(model=)

    res = pipe(text,
               candidate_labels=candidate_labels,
               multi_label=multi_label
               )
    return res


class PageClassifier(Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, page_templates: typing.Callable) -> typing.Callable[[list[str]], dict]:
        """Helps to classify pages based on a list of candidate labels"""

        pts = page_templates()

        @functools.lru_cache
        def _classify_page(candidate_labels: list[str]) -> pd.DataFrame:
            classes = {}
            for page_num, page_text in pts.items():
                res = zero_shot_classifier(page_text, candidate_labels)
                classes[page_num] = dict(
                    page=page_num,
                    **{k: v for k, v in zip(res["labels"], res["scores"])}
                )

            return pd.DataFrame(classes).T.to_dict(orient="records")

        return _classify_page


class TextBlockClassifier(Operator):
    def __init__(self):
        super().__init__()

    def __call__(
            self, text_box_elements: list[pydoxtools.document_base.DocumentElement]
    ) -> list[pydoxtools.document_base.DocumentElement]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_name = "txtblockclassifier"
        model_dir = settings.PDX_MODEL_DIR / model_name
        if not model_dir.exists():
            # TODO: download "any" model that we want from transformerss
            logger.info(f"model {model_name} not found in pydoxtools models, download directly from transformers!")
            model_dir = "xyntopia/tb_classifier"
        # tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512, 'return_tensors': 'pt'}
        # TODO: optionally enable CUDA...
        # TODO: only extract "unique" addresses
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # .to("cuda")
        model = pipeline("text-classification", model=model, tokenizer=tokenizer)
        text = [t.text.strip() for t in text_box_elements]
        res = [model([x], truncation=True, padding=True)[0]["label"] for x in text]
        labeled_text_boxes = [
            dataclasses.replace(t, labels=[label])
            for t, label in zip(text_box_elements, res)
        ]
        return labeled_text_boxes
