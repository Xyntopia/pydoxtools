@cached_property
def ner(self) -> List[Tuple[str, str]]:
    # TODO: add transformers as ner recognition as well:
    #       from transformers import pipeline
    #
    # #ner_pipe = pipeline("ner")
    # #good results = "xlm-roberta-large-finetuned-conll03-english" # large but good
    # #name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english" #small and bad
    # name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    # model = name
    # tokenizer= name
    # ner_pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)
    # TODO: replace the below by self.model
    if self._model_size:
        model_id = nlp_utils.get_spacy_model_id(self.lang, self._model_size)
        nlp = nlp_utils.load_cached_spacy_model(model_id)
    else:
        nlp = nlp_utils.load_cached_spacy_model(self._spacy_model)

    res = nlp_utils.extract_entities_spacy(self.full_text, nlp)
    return res


@cached_property
def grouped_ner(self) -> Dict[str, Any]:
    """Group labels from named entity recognition."""
    groups = group_by([n[::-1] for n in self.ner])
    return groups


@cached_property
def urls(self) -> List[str]:
    urls = nlp_utils.get_urls_from_text(self.full_text)
    return urls


@property
def images(self) -> List:
    return []

@property
def titles(self) -> List[str]:
    return []

@property
def schemadata(self) -> Dict:
    """schema.org data extracted from html meta tags and other metainfos from documents

    TODO: more detailed description of return type"""
    return {}


@property
def product_ids(self) -> Dict[str, str]:
    return {}

@property
def price(self) -> List[str]:
    """if prices are given in the document"""
    return []
