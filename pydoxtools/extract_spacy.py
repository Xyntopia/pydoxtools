import functools
import logging
import subprocess
from typing import Optional, Any

import numpy as np
import spacy
from spacy import Language
from spacy.tokens import Doc, Token, Span

from .document_base import TokenCollection
from .operators_base import Operator

logger = logging.getLogger(__name__)


def download_model(model_id: str):
    # python -m spacy download en_core_web_sm
    return subprocess.call(['python', '-m', 'spacy', 'download', model_id])
    # spacy.load("en_core_web_sm")


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


def get_spacy_model_id(model_language, size="sm") -> Optional[str]:
    """
    size can be: sm, md, lg or trf where "trf" is transformer

    TODO: try to list all spacy models systematically
     """
    if model_language in ['en', 'zh']:
        return f'{model_language}_core_web_{size}'
    elif model_language in ['de', 'fr', 'es', 'ru', 'ja', 'it', 'ca', 'hr', 'da',
                            'nl', 'fi', 'el', 'ko', 'lt', 'mk', 'nb', 'pl', 'pt',
                            'ro', 'ru', 'sv', 'uk']:
        return f'{model_language}_core_news_{size}'
    else:
        return f'xx_sent_ud_sm'


@functools.lru_cache()
def load_cached_spacy_model(model_id: str) -> Language:
    """
    load spacy nlp model and in case of a transformer model add custom vector pipeline...

    we also make sure to cache the model for batch operations on documens.
    """
    try:
        logger.info(f"loading spacy model: {model_id}")
        nlp = spacy.load(model_id)
    except OSError:  # model doesn't seem to be present, yet
        logger.info(f"failed, loading. trying to download spacy model: {model_id}")
        download_model(model_id)
        nlp = spacy.load(model_id)

    if model_id[-3:] == "trf":
        nlp.add_pipe('trf_vectors')

    return nlp


@Language.factory('trf_vectors')
class TrfContextualVectors:
    """
    Spacy pipeline which add transformer vectors to each token based on user hooks.

    https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    https://github.com/explosion/spaCy/discussions/6511
    """

    def __init__(self, nlp: Language, name: str):
        # TODO: we can configure this class for different pooling methods...
        self.name = name
        Doc.set_extension("trf_token_vecs", default=None)

    def __call__(self, sdoc):
        # inject hooks from this class into the pipeline
        if type(sdoc) == str:
            sdoc = self._nlp(sdoc)

        # pre-calculate all vectors for every token:

        # calculate groups for spacy token boundaries in the trf vectors
        vec_idx_splits = np.cumsum(sdoc._.trf_data.align.lengths)
        # get transformer vectors and reshape them into one large continous tensor
        trf_vecs = sdoc._.trf_data.tensors[0].reshape(-1, 768)
        # calculate mapping groups from spacy tokens to transformer vector indices
        vec_idxs = np.split(sdoc._.trf_data.align.dataXd, vec_idx_splits)

        # take sum of mapped transformer vector indices for spacy vectors
        # TOOD: add more pooling methods than just sum...
        #       if we do this we probabyl need to declare a factory function...
        vecs = np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])
        sdoc._.trf_token_vecs = vecs

        sdoc.user_token_hooks["vector"] = self.token_vector
        sdoc.user_span_hooks["vector"] = self.span_vector
        sdoc.user_hooks["vector"] = self.doc_vector
        sdoc.user_token_hooks["has_vector"] = self.has_vector
        sdoc.user_span_hooks["has_vector"] = self.has_vector
        sdoc.user_hooks["has_vector"] = self.has_vector
        # sdoc.user_token_hooks["similarity"] = self.similarity
        # sdoc.user_span_hooks["similarity"] = self.similarity
        # sdoc.user_hooks["similarity"] = self.similarity
        return sdoc

    @functools.lru_cache
    def token_vector(self, token: Token):
        return token.doc._.trf_token_vecs[token.i]

    @functools.lru_cache
    def span_vector(self, span: Span):
        vecs = span.doc._.trf_token_vecs
        return vecs[span.start: span.end].sum(0)

    @functools.lru_cache
    def doc_vector(self, doc: Doc):
        vecs = doc._.trf_token_vecs
        return vecs.sum(0)

    def has_vector(self, token):
        return True


class SpacyOperator(Operator):
    def __call__(
            self,
            full_text: str,
            language: str,
            spacy_model: str,
            model_size: str
    ) -> dict[str, Any]:
        """Load a document using spacy"""
        if spacy_model == "auto":
            nlp_modelid = get_spacy_model_id(language, model_size)
        else:
            nlp_modelid = spacy_model

        spacy_nlp = load_cached_spacy_model(nlp_modelid)
        return dict(
            doc=spacy_nlp(full_text),
            nlp=spacy_nlp
        )
