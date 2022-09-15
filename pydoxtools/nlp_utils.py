#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:29:52 2020

@author: Thomas.Meschede@soprasteria.com

TODO: refactor this... there are a lot of functions and it would
      be great if we could somehow organize them a little better...

TODO: think about how to disribute functions between nlp_utils and
      classifier

TODO: move the inference parts into the "document" class in order to make them document
      agnostic
"""

import functools
import logging
from difflib import SequenceMatcher
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import spacy
import torch
import transformers
from pydantic import BaseModel
from scipy.spatial.distance import pdist, squareform
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from urlextract import URLExtract

from pydoxtools import html_utils
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)

# from transformers import pipeline #for sentiment analysis
compare = sklearn.metrics.pairwise.cosine_similarity


def str_similarity(a, b):
    """
    return string similarity [0..1]
  
    TODO: consider doing this with the levenshtein which is a bit faster
    """
    return SequenceMatcher(None, a, b).ratio()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"using {device}-device for nlp_operations!")

memory = settings.get_memory_cache()
dns_cache_dir = settings.CACHE_DIR_BASE / "urlextract"
dns_cache_dir.mkdir(parents=True, exist_ok=True)
urlextractor = URLExtract(extract_email=True, cache_dns=True, extract_localhost=True,
                          cache_dir=dns_cache_dir)
urlextractor.update_when_older(7)  # updates when list is older that 7 days


# TODO: enhance this with "textrank" algorithms
def self_similarity_matrix(strlist):
    """
    calculates self-similarity of strings using
    pythons difflib in a matrix

    TODO: get rid of pandas dataframe here
    """

    def sim(u, v):
        return str_similarity(u[0].lower(), v[0].lower())

    s = squareform(pdist(strlist.values.reshape(-1, 1), sim))
    return pd.DataFrame(s)  # , columns=strlist, index=strlist)


def get_urls_from_text(text):
    # TODO:  add this to
    urls = urlextractor.find_urls(text, only_unique=False, check_dns=True)
    return urls


def tokenize_windows(txt, tokenizer, win_len=500, overlap=50,
                     max_len=510, add_special_tokens=True):
    # token_ids = tokenizer.encode(txt,add_special_tokens=False)
    toktxt = tokenizer.tokenize(txt)
    # tokenizer.convert_tokens_to_ids(toktxt)

    tk_num = len(toktxt)
    if tk_num < max_len:
        toktxt = tokenizer.encode(txt)
        if add_special_tokens:
            return [toktxt], toktxt
        else:
            return [toktxt[1:-1]], toktxt
    else:
        step = int(win_len - overlap)

        steps = list(range(0, tk_num, step))
        tok_wins = [toktxt[idx:idx + win_len] for idx in steps]

        # add [CLS] and [SEP] tokens to the windows and encode into ids
        cls_tok = tokenizer.special_tokens_map['cls_token']
        sep_tok = tokenizer.special_tokens_map['sep_token']
        if add_special_tokens:
            tok_wins = [[cls_tok] + win + [sep_tok] for win in tok_wins]
        tok_wins_ids = [tokenizer.convert_tokens_to_ids(win) for win in tok_wins]
        return tok_wins_ids, toktxt


def transform_to_contextual_embeddings(input_ids_t, model, tokenizer=None, lang=False):
    # for one sentence all ids are "1" for two, the first sentence gets "0"
    input_ids_t = torch.tensor([input_ids_t]).to(device)
    segments_ids_t = torch.tensor([[1] * input_ids_t.shape[1]]).to(device)

    # create language IDs
    if lang:
        language_id = tokenizer.lang2id[lang]  # 0
        langs_t = torch.tensor([language_id] * input_ids_t.shape[1])  # torch.tensor([0, 0, 0, ..., 0])
        # We reshape it to be of size (batch_size, sequence_length)
        langs_t = langs_t.view(1, -1)  # is

    # Predict hidden states features for each layer
    with torch.no_grad():
        if lang:  # if XLM cross lingual models
            encoded_layers = model(input_ids_t, langs=langs_t)
            wvecs_out = encoded_layers[0].detach().cpu().numpy()
        else:  # if using BERT-likes/multilingual
            encoded_layers = model(input_ids_t, segments_ids_t)
            # distilbert:
            # remove [CLS] and [SEP] tokens
            wvecs_out = encoded_layers[0][0].cpu().numpy()[1:-1]
            # wvecs_out = encoded_layers[1][-1][0].numpy()[1:-1]
            # this one is for "normal" BERT:
            # wvecs_out = encoded_layers[2][-1][0].numpy()[1:-1]
            # wvecs_out = encoded_layers[2][-1][0].numpy()[1:-1]
    return wvecs_out


# TODO: prepare for long texts to do the tokenization in batches
# otherwise well run out of memory :(
def longtxt_word_embeddings_fullword_only_static_embeddings(txt, tokenizer):
    """
    generate whole-word embeddings (without pseudo-syllables)
    using only transformers tokenizer without
    model.
    """
    vs, toktxt = get_embeddings(txt, tokenizer)
    return fullword_embeddings(toktxt, vs)


def longtxt_embeddings_fullword(txt, model, tokenizer):
    """
    generate whole-word embeddings (without pseudo-syllables)
    using transformers models.
    """
    vs, toktxt = longtxt_embeddings(txt, model, tokenizer)
    return fullword_embeddings(toktxt, vs)


def longtxt_embeddings(txt, model, tokenizer,
                       pooling=None,
                       overlap=50,
                       longtextcap=True,
                       status_bar=False):
    """
    generate wordpiece embeddings (pseudo-syllables) using transformer model
    and text windowing. The individual windows are stitched
    back together at the and by averaging their values

    TODO: add option to cancel embeddings generation afte a certain
          number of windows to make sure it finishes in a guaranteed time
    """
    tok_wins, toktxt = tokenize_windows(txt, tokenizer=tokenizer,
                                        overlap=overlap)
    if longtextcap:
        tok_wins = tok_wins[:100]

    if status_bar:  # only use tqdm for "long lasting" transformations
        vec_wins = [transform_to_contextual_embeddings(win, model=model)
                    for win in tqdm(tok_wins)]
    else:
        vec_wins = [transform_to_contextual_embeddings(win, model=model)
                    for win in tok_wins]
    # pd.DataFrame(vec_wins).shapeFalse

    if len(vec_wins) == 1:
        vecs = vec_wins[0]
    else:
        win_len = vec_wins[0].shape[0]
        step = int(win_len - overlap)
        vecs = vec_wins[0][:step]
        for i in range(len(vec_wins) - 1):
            # average overlapping vectors of current and next window
            nxtvecs = (vec_wins[i][step:] + vec_wins[i + 1][:overlap]) / 2
            vecs = np.vstack((vecs, nxtvecs))
            vecs = np.vstack((vecs, vec_wins[i + 1][overlap:step]))
        # vecs = np.vstack((vecs,vec_wins[-1][step:]))

    if pooling == None:
        return vecs, toktxt
    else:
        return pooling(vecs, axis=0), toktxt


# old name: create_cross_lingual_embeddings
def create_cross_lingual_contextual_embeddings(txt, model, tokenizer, lang=False):
    # Map the token strings to their vocabulary indeces.
    # indexed_tokens = tokenizer.convert_tokens_to_ids(toktxt)
    input_ids_t = torch.tensor([tokenizer.encode(txt)])

    # for one sentence all ids are "1" for two, the first sentence gets "0"
    segments_ids_t = torch.tensor([[1] * input_ids_t.shape[1]])

    # create language IDs
    if lang:
        language_id = tokenizer.lang2id[lang]  # 0
        langs_t = torch.tensor([language_id] * input_ids_t.shape[1])  # torch.tensor([0, 0, 0, ..., 0])
        # We reshape it to be of size (batch_size, sequence_length)
        langs_t = langs_t.view(1, -1)  # is

    # Predict hidden states features for each layer
    with torch.no_grad():
        if lang:  # if XLM cross lingual models
            encoded_layers = model(input_ids_t, langs=langs_t)
            wvecs_out = encoded_layers[0].detach().numpy()
        else:  # if using BERT-likes
            encoded_layers = model(input_ids_t, segments_ids_t)
            wvecs_out = encoded_layers[0].numpy()

    return wvecs_out


def cos_compare(X, Y):
    """TODO: replace cos_compare with cos_similarty based on numpy..."""
    return sk.metrics.pairwise.cosine_similarity(X, Y)


def cos_similarity(a, b):
    """compare to vectors using cosine similarity"""
    IaI = np.sqrt((a * a).sum(1))
    IbI = np.sqrt((b * b).sum(1))
    ab = (a * b).sum(1)
    d = ab / (IaI * IbI)
    return d


"""
def build_pipe(X, params):
    classifiers=[
        sk.linear_model.LogisticRegression(max_iter=200,
                                            verbose=1,
                                            n_jobs=-1),
        sk.linear_model.LogisticRegressionCV(),
        sk.linear_model.RidgeClassifier(),
        sk.svm.LinearSVC(),
        sk.linear_model.SGDClassifier(),
        sk.linear_model.PassiveAggressiveClassifier(),
        sk.naive_bayes.BernoulliNB(alpha=.01), 
        sk.naive_bayes.ComplementNB(alpha=.01), 
        sk.naive_bayes.MultinomialNB(alpha=.01),
        sk.neighbors.KNeighborsClassifier(),
        sk.neighbors.NearestCentroid(),
        sk.ensemble.RandomForestClassifier(),
        ]
    
    classifier = classifiers[params['classifier']]
    logger.info(f"using classifier: {classifier}")

    txt_tf = sk.pipeline.make_pipeline(
            sk.preprocessing.FunctionTransformer(
                    select_long_description,
                    validate=False), #has to be "False" to allow strings
            TfidfVectorizer(
                tokenizer = nc_tokenizer,
                max_df = params["maxdf"],
                max_features=params['maxfeat'],
                ngram_range=params["ngram_range"],
                preprocessor=None)
            )

    #txt_tf.fit(X_train,y_train)
    #tfidf_vector
    logger.info("selecting columns")
    select_cols = select_feature_columns(X,max_unique_cat=params['max_unique_cat'])
    
    cat_tf = skl.pipeline.make_pipeline(
            skl.preprocessing.FunctionTransformer(
                    select_columns,validate=False,
                    kw_args={'columns':select_cols}),
            skl.impute.SimpleImputer(strategy='constant', fill_value='missing'),
            skl.preprocessing.OneHotEncoder(handle_unknown='ignore'))
    
    pipe = skl.pipeline.make_pipeline(
            skl.pipeline.make_union(txt_tf, cat_tf),
            #skl.decomposition.TruncatedSVD(n_components=1000),
            classifier)

    return pipe
"""


@functools.lru_cache()
def get_bert_vocabulary():
    model, tokenizer = load_models()
    # return and transform embeddings into numpy array
    return model.embeddings.word_embeddings.weight.detach().numpy()


def get_spacy_model_id(model_language, size="sm") -> Optional[str]:
    """size can be: sm, md, lg or trf where "trf" is transformer """
    if model_language == 'en':
        return f'en_core_web_{size}'
    elif model_language == 'de':
        return f'de_core_news_{size}'
    else:
        None


@functools.lru_cache()
def load_cached_spacy_model(model_id: str) -> Language:
    """load spacy nlp model and in case of a transformer model add custom vector pipeline..."""
    nlp = spacy.load(model_id)
    if model_id[-3:] == "trf":
        nlp.add_pipe('trf_vectors')

    return nlp


def generate_spacy_model_id_list(options: List[str] = None):
    model_names = [
        'xx_ent_wiki_sm', 'en_core_web_md', 'de_core_news_md',
        'en_core_web_sm', 'de_core_news_sm'
    ]
    if options:
        if 'lg' in options:
            model_names += ['en_core_web_lg', 'de_core_news_lg']
        if 'trf' in options:
            model_names += ['en_core_web_trf', 'de_dep_news_trf']

    return model_names


def download_spacy_nlp_models(options: List[str], dl_path=None):
    """download models and other necessary stuff
    if we need more models, we can find them here:

    https://spacy.io/usage/models#download-manual

    we can use this function to pre-download & initialize
    our models in a dockerfile like this:

        python -c 'from pydoxtools import nlp_utils; nlp_utils.download_nlp_models(["trf","md"])'

    """
    model_names = generate_spacy_model_id_list(options)

    # https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.4.0/xx_ent_wiki_sm-3.4.0-py3-none-any.whl
    for model_id in model_names:
        version = "3.4.0"
        # spacy.about.__version__  doesn't work ..  probably because models re only availabe with major versions
        # meaning 3.4.0 instead of 3.4.1
        model_version = f"{model_id}-{version}"
        url = f"{spacy.about.__download_url__}/{model_version}/{model_version}-py3-none-any.whl"
        """
        try:
            nlp=spacy.load(model_id)
            logger.info(f"model {model_id} is already installed!")
        except IOError:
            spacy.cli.download(model_id)
            nlp = spacy.load(model_id)
        #(settings.MODEL_DIR/"spacy").mkdir(parents=True, exist_ok=True)
        #nlp.to_disk(settings.MODEL_DIR/"spacy"/model_id)
        """
        import subprocess
        # pip download -d # would lso be an option...
        # print(subprocess.check_output(['pip', 'install', url]))
        print(f"dowloading to: {dl_path}")
        if dl_path:
            subprocess.call(['pip', 'download', '--no-deps', '-d', dl_path, url])
        else:
            subprocess.call(['pip', 'install', '--no-deps', url])

        # pip download --no-deps https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.4.0/xx_ent_wiki_sm-3.4.0-py3-none-any.whl


def reset_models():
    """clear models from memory"""
    load_models.cache_clear()
    load_tokenizer.cache_clear()


def veclengths(x):
    return np.sqrt((x * x).sum(axis=1))


def maxlens(x):
    return np.max(x, axis=1)


def vecseq_similarity(vs, search_vec):
    return cos_compare(vs, [search_vec])


@functools.lru_cache()
def get_vocabulary(kind="distilbert"):
    """make sure the vocabulary only gets loaded once
    TODO: implement more vocabularies"""
    logger.info("loading BERT vocabulary")
    return get_bert_vocabulary()


def get_embeddings(txt, tokenizer):
    """
    generate word-piece embeddings (pseudo-syllables)
    using only transformers tokenizer without
    model.
    """
    txttok = tokenizer.tokenize(txt)
    tok_ids = tokenizer.convert_tokens_to_ids(txttok)
    tok_vecs = get_vocabulary()[tok_ids]
    return tok_vecs, txttok


def fullword_embeddings(toktxt, vs):
    """
    get embeddings for entire words by sowing wordpieces back together.

    Parameters
    ----------
    toktxt : tokenized text
        DESCRIPTION.
    vs : word piece vectors
        DESCRIPTION.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        return full word-tokens and vectors

    """
    # average embedding vectors for entire words:
    emb_map = list(zip(toktxt, vs))

    syl_sep = ""  # could also be "-" for example

    newtoks = []
    newvs = []
    cur_word = ""
    cur_vec = []
    for tok, v in emb_map:
        if len(tok) >= 3:
            if tok[:2] == '##':
                cur_word += tok
                cur_vec += [v]
                continue
        newtoks += [cur_word.replace("##", syl_sep)]
        # TODO:  replace np.mean with np.sum here and below?
        newvs += [np.mean(cur_vec, axis=0)]
        cur_vec = [v]
        cur_word = tok

    newtoks += [cur_word.replace("##", syl_sep)]
    newvs += [np.mean(cur_vec, axis=0)]

    # newtoks = [txt.encode("windows-1252").decode("utf-8")
    #               for txt in newtoks[1:]]
    return np.array(newvs[1:]), np.array(newtoks[1:])


def top_search_results(toktxt, match, num=10):
    """
    returns:
        best matching tokens, their ids and corresponding
        scores
    """
    toktxt = np.array(toktxt)
    idxs = np.argpartition(match.flatten(), -num)[-num:][::-1]
    idxs = idxs[np.argsort(match[idxs][:, 0])][::-1]
    return toktxt[idxs], idxs, match[idxs]


def get_max_word_similarity(vs, searchstring, model, tokenizer):
    sv, _ = get_embeddings(searchstring, model, tokenizer)
    match = vecseq_similarity(vs, sv.mean(axis=0))
    return match.max()


def search(toktxt, vs, searchstring, model, tokenizer, num=1):
    """
    returns:
        top tokens, token ids, correponding scores, all token scores
    """
    # sv, _ = longtxt_embeddings(search_word,model,tokenizer)
    sv, _ = get_embeddings(searchstring, tokenizer)

    match = vecseq_similarity(vs, sv.mean(axis=0))
    return top_search_results(toktxt, match, num=num) + (match,)


def get_keywords():
    raise NotImplementedError
    # justsome ideas in the following
    # TODO: generate "vs" using transform_to_contextual_embeddings
    # TODO: maybe to a ANN search with the vocabulary from BERT?
    similarity = nlpu.vecseq_similarity(vs, sentvec)
    wordranks = pd.DataFrame(zip(similarity, tokwords),
                             columns=['similarity', 'words'])
    wordranks['importance'] = importance * wordranks['similarity']
    # colhtml = html_utils.color_text(tokwords, similarity)
    # oib = html_utils.oib
    # oib(colhtml)


# def topic_similarity(model):

# these urls were selected, because they have particularly long webpages
# to slow down the classifiers etc...
example_urls = [
    "https://www.newark.com/c/passive-components/resistors-fixed-value",
    "https://www.newark.com/c/test-measurement/test-equipment-accessories",
    "https://www.newark.com/c/enclosures-racks-cabinets/enclosures-boxes-cases",
    "https://www.newark.com/c/circuit-protection/tvs-transient-voltage-suppressors",
    "https://chicagodist.com/collections/pololu",
    "https://www.newark.com/c/semiconductors-discretes/transistors",
    "https://chicagodist.com/collections/all-in-stock-items",
    "https://buyzero.de/products/raspberry-pi-4-model-b?variant=28034033287270",
    "https://buyzero.de/products/raspberry-pi-4-model-b?variant=28034033090662",
    "https://buyzero.de/products/raspberry-pi-4-model-b?variant=28034034008166",
]


def string_embeddings(text, method="fast"):
    """
    this method converts a text of arbitrary length into
    a vector.
    """
    if method == "fast":
        tokenizer, _ = load_tokenizer()
        vs, toktxt = get_embeddings(text, tokenizer)
        vs = vs.mean(axis=0)
    elif method == "slow":
        model, tokenizer = load_models()
        vs, toktxt = longtxt_embeddings(
            text, model, tokenizer, np.mean)
    return vs


def page2vec(page_str, url=None, method="slow"):
    """
    TODO: use the document class for this....

    calculate a fingerprint from any arbitrary webpage
    
    TODO: include potential incoming links in fingerprint.
        Those would have to be search independently
    
    TODO: include more information into vectorization such as
    - tag-density
    - link-density
    - nomalized pagelength
    
    - which tags
    - structure of html
    - screenshot of page
    """
    # +length = len(html)
    # length = len(" ".join(html.split()))
    if method == "no_embeddings":
        vs = [
            len(url),
            len(page_str),
        ]
    elif method in ["slow", "fast"]:
        try:
            # TODO: can we somehow move html_utils out of this file?
            text_short = html_utils.get_pure_html_text(page_str)
            vs = string_embeddings(text_short, method)
        except:
            logger.exception(f"can not convert: {url}")
            return None

    vectorization = vs
    return vectorization


# TODO implement this function for reinforcement link following
def link2vec(source_html, source_url, link_context, link_url):
    raise NotImplementedError()


def topic_similarity(html, topic, method="slow"):
    """
    compare similarity of a given html page
    with a certain topic.
    
    The result has a range between 0..1 but
    usually results will be in the range
    [0..
     TODO: (~0.01)]
    
    TODO: make a more "finegrained analysis" 
    """
    vs = page2vec(html, method)

    model, tokenizer = load_models()
    sv, _ = longtxt_embeddings(topic, model, tokenizer, np.mean)
    # sv, _ = get_embeddings(searchstring,tokenizer)

    similarity = cos_compare([vs], [sv])
    # match = vecseq_similarity(vs,sv.mean(axis=0))

    return similarity


def extract_entities_spacy(text, nlp):
    """extract entitis from text
    
    "nlp" can either be a spacy nlp object or a transformers pipeline

    # TODO: move this into our "doc" class
    """
    # TODO: also enable transformers pipelines like this:

    # from transformers import pipeline

    # ner_pipe = pipeline("ner")
    # good results = "xlm-roberta-large-finetuned-conll03-english" # large but good
    # name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english" #small and bad
    # name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    # model = name
    # tokenizer= name
    # ner_pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)

    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# TODO: we need to move this into the nlp_context class
#      and provide the use of the models with a timeout
#      or a "with" context or something similar...

# @functools.lru_cache()


def convert_ids_to_string(tokenizer, ids):
    a = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(a)


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


# we are creating a factory here as our tranformers vector calculation is stateful
# and we need a specific class for this..
# @Language.factory("my_component", default_config={"some_setting": True})
# def my_component(nlp, name, some_setting: bool):
#    return MyComponent(some_setting=some_setting)
class NLPContext(BaseModel):
    # doesn't work with trasformers yet because AutoTokenizer/Model
    # are converted into the respective model classes which don't inherit from Autotokenizer...
    # TODO: find a potential base class?
    # TODO: generalize this class with nlp_utils loading models...
    tokenizer: Any  # transformers.AutoTokenizer
    model: Any  # transformers.AutoModel
    capabilities: set[str] = []  # model capabilities e.g. "qam"  or "ner"

    class Config:
        # we need this as pydantic doesn't have validators for transformers models
        arbitrary_types_allowed = True


@functools.lru_cache(maxsize=32)
def load_tokenizer(model_name='distilbert-base-multilingual-cased'):
    logger.info("load_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, get_vocabulary()


# TODO: merge this function with
@functools.lru_cache()
def load_models(model_name: str = 'distilbert-base-multilingual-cased'):
    logger.info(f"load model on device: {device}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    return model, tokenizer


@functools.lru_cache()
def QandAmodels(model_id: str):
    # TODO: only load model "id" and use that id
    #        with transformers AutoModel etc...
    logger.info(f"loading Q & A model and tokenizer {model_id}")
    # model, tokenizer = load_models(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    logger.info(f"finished loading Q & A models... {model_id}")
    return NLPContext(tokenizer=tokenizer, model=model)
