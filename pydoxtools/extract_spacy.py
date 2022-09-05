from functools import cached_property

import spacy

import nlp_utils
from document import Extractor


class spacy_extractor(Extractor):
    def __init__(self, spacy_model: str = "",
                 model_size: str = "",
                 ):
        self._spacy_model = spacy_model
        self._model_size = model_size

    @cached_property
    def spacy_model_id(self) -> str:
        if self._spacy_model:
            nlpmodelid = self._spacy_model
        elif self._model_size:
            nlpmodelid = nlp_utils.get_spacy_model_id(self.lang, self._model_size)
        else:
            nlpmodelid = "xx_ent_wiki_sm"

        return nlpmodelid

    @cached_property
    def spacy_nlp(self) -> spacy.Language:
        nlp = nlp_utils.load_cached_spacy_model(self.spacy_model_id)
        return nlp

    @cached_property
    def spacy_doc(self) -> spacy.tokens.Doc:
        return self.spacy_nlp(self.full_text)

    @property
    def vectors(self):
        # TODO: also use vectors without context! (using BERT embeddings for example...)
        # TODO: this is the "old" method without spacy... we might consider
        #       to use this method again in order to leverage more huggingface models such as the multilingual one...
        #       the problem is a little bit, that "custom" huggingface mdoels don't 100% align with
        #       the spacy tokens...
        # model,tokenizer = nlp_utils.load_models()
        # v = nlp_utils.longtxt_embeddings_fullword(doc.full_text, model, tokenizer)
        # v[1]
        # pd.DataFrame(list(zip(v[1],df['text'])), columns=['trf','tok']).head(50)
        return self.spacy_doc._.trf_token_vecs

    @property
    def trf_embeddings(self):
        trfc = self.spacy_nlp.components[0][1]
        return trfc.model.transformer.embeddings.word_embeddings

    @cached_property
    def noun_chunks(self) -> list[TokenCollection]:
        token_list = []
        for nc in self.spacy_doc.noun_chunks:
            tc = TokenCollection([t for t in nc if t.pos_ not in ["DET", "SPACE", "PRON"]])
            if len(tc) > 0:
                token_list.append(tc)
        # filter = ["DET"]
        return token_list

    @cached_property
    def sents(self) -> list[spacy.tokens.Span]:
        return list(self.spacy_doc.sents)
