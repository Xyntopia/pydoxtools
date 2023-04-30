#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:29:52 2020

@author: Thomas.Meschede@soprasteria.com

we accumulate most of our NLP functions here for the rest of the library
as we need to keep models in memory for other functions to use
thats why there is functools.lru_cache spread around everywhere here.
Otherwise NLP models would get reloaded everywhere which we
don't want in our pipelines.

TODO: refactor this... there are a lot of functions and it would
      be great if we could somehow organize them a little better...

TODO: think about how to disribute functions between nlp_utils and
      classifier
"""

import functools
import logging
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import torch
from pydantic import BaseModel
from scipy.spatial.distance import pdist, squareform
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


def get_embeddings(txt, tokenizer):
    """
    generate word-piece embeddings (pseudo-syllables)
    using only transformers tokenizer without
    model.
    """
    txttok = tokenizer.tokenize(txt)
    tok_ids = tokenizer.convert_tokens_to_ids(txttok)
    tok_vecs = get_vocabulary(tokenizer.name_or_path)[tok_ids]
    return tok_vecs, txttok


def tokenize_windows(txt, tokenizer, win_len=500, overlap=50,
                     max_len=510, add_special_tokens=True):
    """
    Tokenize a given text into a series of overlapping windows with special tokens.

    This function tokenizes a given text into a series of overlapping windows,
    adds special tokens ([CLS], [SEP]) if required, and converts the tokens into
    token IDs. The function accepts parameters for the window length, overlap between
    consecutive windows, maximum tokenization length, and a flag to add special tokens.
    It returns the tokenized windows with token IDs and the tokenized text without
    token IDs.

    Args:
        txt (str): The input text to be tokenized.
        tokenizer (Tokenizer): The tokenizer instance to be used for tokenization.
        win_len (int, optional): The length of each token window. Default is 500.
        overlap (int, optional): The number of overlapping tokens between consecutive windows. Default is 50.
        max_len (int, optional): The maximum length for tokenization. Default is 510.
        add_special_tokens (bool, optional): Whether to add special tokens ([CLS], [SEP]) to the tokenized text. Default is True.

    Returns:
        list: A list of lists containing the tokenized windows with token IDs.
        list: The tokenized text without token IDs.

    Example:

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        txt = "This is a sample text."
        tok_wins_ids, toktxt = tokenize_windows(txt, tokenizer)

    """
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


def longtxt_fullword_embeddings_only_lookup_table(txt, tokenizer):
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


def longtxt_embeddings(
        txt, model, tokenizer,
        pooling=None,
        overlap=50,
        longtextcap=True,
        status_bar=False
):
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
def get_vocabulary(model_id: str):
    """make sure the vocabulary only gets loaded once
    TODO: implement more vocabularies"""
    logger.info(f"loading vocabulary from {model_id}")
    model, _ = load_models(model_id)
    # return and transform embeddings into numpy array
    return model.embeddings.word_embeddings.weight.detach().numpy()


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
    sv, _ = get_embeddings(searchstring, tokenizer)
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


def calculate_string_embeddings(text: str, model_id: str, only_tokenizer: bool):
    """
    this method converts a text of arbitrary length into
    a vector.
    """
    tokenizer = load_tokenizer(model_id)
    if only_tokenizer:
        vs, toktxt = get_embeddings(text, tokenizer)
        vs = vs.mean(axis=0)
        return vs
    else:
        model = load_models()
        vs, toktxt = longtxt_embeddings(text, model, tokenizer, np.mean)
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
            vs = calculate_string_embeddings(text_short, method)
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

    model_id = settings.PDXT_STANDARD_TOKENIZER
    tokenizers = load_tokenizer(model_id)
    model = load_models(model_id)
    sv, _ = longtxt_embeddings(topic, model, tokenizer, np.mean)
    # sv, _ = get_embeddings(searchstring,tokenizer)

    similarity = cos_compare([vs], [sv])
    # match = vecseq_similarity(vs,sv.mean(axis=0))

    return similarity


# TODO: we need to move this into the nlp_context class
#      and provide the use of the models with a timeout
#      or a "with" context or something similar...

# @functools.lru_cache()


def convert_ids_to_string(tokenizer, ids):
    a = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(a)


# TODO: get rid of this...
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
def load_tokenizer(model_name):
    logger.info("load_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# TODO: merge this function with
@functools.lru_cache()
def load_models(model_id: str):
    logger.info(f"load model on device: {device}")
    tokenizer = load_tokenizer(model_id)
    # model = AutoModelForQuestionAnswering.from_pretrained(model_id, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    model.to(device)
    model.eval()
    return model, tokenizer


@functools.lru_cache()
def load_pipeline(pipeline_type: str, model_id: str):
    from transformers import pipeline
    pipeline_instance = pipeline(pipeline_type, model=model_id)
    return pipeline_instance


@functools.lru_cache()
def QandAmodels(model_id: str):
    # TODO: only load model "id" and use that id
    #        with transformers AutoModel etc...
    logger.info(f"loading Q & A model and tokenizer {model_id}")
    # model, tokenizer = load_models(model_id)
    # TODO: use load_tokenizer function for this
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    logger.info(f"finished loading Q & A models... {model_id}")
    return NLPContext(tokenizer=tokenizer, model=model)


def summarize_long_text(
        long_text: str,
        model_id: str,
        token_overlap=50,
        max_len=200,
):
    pipeline = load_pipeline("summarization", model_id=model_id)
    model, tokenizer = pipeline.model, pipeline.tokenizer
    max_input_tokens = pipeline.model.config.max_position_embeddings

    def summarize_chunks(text) -> (str, int):
        inputs = tokenizer(text, return_tensors="pt").input_ids

        # Split the input into smaller chunks if it's longer than the maximum number of tokens
        input_chunks = []
        for i in range(0, inputs.shape[1], max_input_tokens - token_overlap):
            # Subtracting token_overlap to account for potential token overlap
            input_chunks.append(inputs[0][i:i + max_input_tokens])

        # Summarize each chunk
        summarized_chunks = []
        for chunk in input_chunks:
            # Convert tensor to list and remove padding tokens
            chunk = list(filter(lambda x: x != tokenizer.pad_token_id, chunk.tolist()))
            # Generate summary for the current chunk
            summary_ids = model.generate(torch.tensor([chunk]), max_new_tokens=max_len, do_sample=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summarized_chunks.append(summary)

        # Combine the summarized chunks
        return " ".join(summarized_chunks), len(input_chunks)

    summary, chunk_num = summarize_chunks(long_text)
    while chunk_num > 1:
        summary, chunk_num = summarize_chunks(summary)

    return summary
