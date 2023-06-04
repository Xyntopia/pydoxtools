from __future__ import annotations  # this is so, that we can use python3.10 annotations..

from typing import Any

import pandas as pd
from urlextract import URLExtract

from pydoxtools.operators_base import Operator
from pydoxtools.settings import settings

dns_cache_dir = settings.PDX_CACHE_DIR_BASE / "urlextract"
dns_cache_dir.mkdir(parents=True, exist_ok=True)
urlextractor = URLExtract(extract_email=True, cache_dns=True, extract_localhost=True,
                          cache_dir=dns_cache_dir)
urlextractor.update_when_older(7)  # updates when list is older that 7 days


class EntityExtractor(Operator):
    def __call__(self, spacy_doc) -> dict[str, list[str]]:
        """TODO: add more entity extraction algorithms (e.g. hugginface)"""
        # TODO: add transformers as ner recognition as well:
        #       from transformers import pipeline
        # #ner_pipe = pipeline("ner")
        # #good results = "xlm-roberta-large-finetuned-conll03-english" # large but good
        # #name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english" #small and bad
        # name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
        # model = name
        # tokenizer= name
        # ner_pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)

        entities = []
        spacy_entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]

        entities.extend(spacy_entities)
        entities = pd.DataFrame(entities, columns=["entity", "label"])
        entities.entity = entities.entity.str.strip()
        # entity_groups = entities.groupby('label').apply(lambda x: x.entity.value_counts().to_dict())
        entity_groups = entities.groupby('label').entity.apply(list)

        return dict(
            entities=entity_groups.to_dict()
        )


def grouped_ner(self) -> dict[str, Any]:
    """Group labels from named entity recognition."""
    groups = group_by([n[::-1] for n in self.ner])
    return groups


def urls(self) -> list[str]:
    urls = urlextractor.find_urls(text, only_unique=False, check_dns=True)
    return urls


@property
def images(self) -> list:
    return []


@property
def titles(self) -> list[str]:
    return []


@property
def schemadata(self) -> dict:
    """schema.org data extracted from html meta tags and other metainfos from documents

    TODO: more detailed description of return type"""
    return {}


@property
def product_ids(self) -> dict[str, str]:
    return {}


@property
def price(self) -> dict[str]:
    """if prices are given in the document"""
    return []
