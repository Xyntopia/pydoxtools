import logging

import spacy

from pydoxtools import nlp_utils
from .document_base import Extractor, TokenCollection

logger = logging.getLogger(__name__)


def download_space_models():
    """"""
    # !/usr/bin/env python3
    logger.info("downloading some standard spacy models!")
    model_names = [
        'xx_ent_wiki_sm', 'en_core_web_md', 'de_core_news_md',
        'en_core_web_sm', 'de_core_news_sm',
        # 'en_core_web_lg', 'de_core_news_lg'
        # 'en_core_web_trf', 'de_dep_news_trf'
    ]
    import subprocess
    for model_id in model_names:
        # python -m spacy download en_core_web_sm
        subprocess.call(['python', '-m', 'spacy', 'download', model_id])
        # spacy.load("en_core_web_sm")


class SpacyExtractor(Extractor):
    def __init__(
            self,
            model_size: str = "sm",
            model_language: str = "auto",
            spacy_model="xx_ent_wiki_sm"
    ):
        """
        model_size: if model_language=="auto" we also need to set our model_size

        TODO: add a "HuggingfaceExtractor" with similar structure

        """
        super().__init__()
        self._spacy_model = spacy_model
        self._model_size = model_size
        self._model_language = model_language

    def __call__(self, full_text: str, language: str = "auto") -> spacy.tokens.Doc:
        if self._model_language == "auto":
            nlp_modelid = nlp_utils.get_spacy_model_id(language, self._model_size)
        else:
            nlp_modelid = self._spacy_model

        spacy_nlp = nlp_utils.load_cached_spacy_model(nlp_modelid)
        return dict(
            doc=spacy_nlp(full_text),
            nlp=spacy_nlp
        )


def extract_noun_chunks(spacy_doc) -> list[TokenCollection]:
    token_list = []
    for nc in spacy_doc.noun_chunks:
        tc = TokenCollection([t for t in nc if t.pos_ not in ["DET", "SPACE", "PRON"]])
        if len(tc) > 0:
            token_list.append(tc)
    # filter = ["DET"]
    return token_list


def extract_spacy_token_vecs(spacy_doc):
    if spacy_doc.has_vector:
        return spacy_doc.tensor
    else:
        return spacy_doc._.trf_token_vecs


def get_spacy_embeddings(spacy_nlp):
    try:
        return spacy_nlp.components[0][1].model.transformer.embeddings.word_embeddings
    except AttributeError:
        # not sure how to get embeddings from here:      t2v = spacy_nlp.components[0][1]
        return []
