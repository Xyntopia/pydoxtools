#!/usr/bin/env python3
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

from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import logging
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering

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


@functools.lru_cache(maxsize=32)
def load_tokenizer(model_id):
    logger.info("load_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


@functools.lru_cache
def load_model(model_id: str) -> Any:
    logger.info(f"load model {model_id} on device: {device}")
    # model = AutoModelForQuestionAnswering.from_pretrained(model_id, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    model.to(device)
    model.eval()
    return model


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


def get_tokenizer_only_embeddings(txt: str, model_id: str):
    """
    generate word-piece embeddings (pseudo-syllables)
    using only transformers tokenizer without
    model.
    """
    tokenizer = load_tokenizer(model_id=model_id)
    txttok = tokenizer.tokenize(txt)
    tok_ids = tokenizer.convert_tokens_to_ids(txttok)
    tok_vecs = get_vocabulary(tokenizer.name_or_path)[tok_ids]
    return tok_vecs, txttok


def tokenize_windows(txt, model_id: str, win_len: int = 500, overlap_ratio: float = 0.1,
                     max_len: int = 510, add_special_tokens: bool = True):
    """
    Tokenize text into overlapping windows with special tokens.

    Given an input text, this function tokenizes it into a series of overlapping windows,
    adds special tokens ([CLS], [SEP]) if required, and converts the tokens into
    token IDs. It returns the tokenized windows with token IDs and the tokenized text without
    token IDs.

    Args:
        txt (str): The input text to be tokenized.
        model_id (str): The model ID for the tokenizer to be used for tokenization.
        win_len (int, optional): The length of each token window. Default is 500.
        overlap_ratio (float, optional): The ratio of overlapping tokens between consecutive windows. Default is 0.1.
        max_len (int, optional): The maximum length for tokenization. Default is 510.
        add_special_tokens (bool, optional): Whether to add special tokens ([CLS], [SEP]) to the tokenized text.
            Default is True.

    Returns:
        Tuple[List[List[int]], List[str]]: A tuple containing a list of lists with the tokenized windows' token IDs
            and the tokenized text without token IDs.

    Example:

        txt = "This is a sample text."
        tok_wins_ids, toktxt = tokenize_windows(txt, "bert-base-uncased")

    """
    overlap = int(max_len * overlap_ratio)
    tokenizer = load_tokenizer(model_id)
    toktxt = tokenizer.tokenize(txt)

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


def transform_to_contextual_embeddings(input_ids_t, model_id: str, tokenizer=None, lang=False):
    """
    Create contextual embeddings using a huggingface model
    """
    # for one sentence all ids are "1" for two, the first sentence gets "0"
    input_ids_t = torch.tensor([input_ids_t]).to(device)
    segments_ids_t = torch.tensor([[1] * input_ids_t.shape[1]]).to(device)

    # create language IDs
    if lang:
        language_id = tokenizer.lang2id[lang]  # 0
        langs_t = torch.tensor([language_id] * input_ids_t.shape[1])  # torch.tensor([0, 0, 0, ..., 0])
        # We reshape it to be of size (batch_size, sequence_length)
        langs_t = langs_t.view(1, -1)  # is

    model = load_model(model_id=model_id)

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


def longtxt_fullword_embeddings_only_lookup_table(txt: str, model_id: str):
    """
    generate whole-word embeddings (without pseudo-syllables)
    using only transformers tokenizer without
    model.
    """
    vs, toktxt = get_tokenizer_only_embeddings(txt, model_id=model_id)
    return fullword_embeddings(toktxt, vs)


def longtxt_embeddings_fullword(txt, model_id: str):
    """
    generate whole-word embeddings (without pseudo-syllables)
    using transformers models.
    """
    vs, toktxt = longtxt_embeddings(txt, model_id)
    return fullword_embeddings(toktxt, vs)


def longtxt_embeddings(
        txt: str, model_id: str,
        overlap_ratio: float = 0.1, longtextcap: bool = True,
        status_bar: bool = False
) -> tuple[np.ndarray, list[str]]:
    """
    Generate wordpiece embeddings using a transformer model and text windowing.

    This function generates wordpiece embeddings (pseudo-syllables) using a transformer model
    and text windowing. The individual windows are stitched back together by averaging their
    values in the overlapped areas.

    Args:
        txt (str): Input text for generating embeddings.
        model_id (str): Model ID for the transformer model to be used.
        overlap_ratio (float, optional): Ratio of overlapping tokens between consecutive windows. Default is 0.1.
        longtextcap (bool, optional): Limit the number of tokenized windows to 100. Default is True.
        status_bar (bool, optional): Display a progress bar for long-lasting transformations. Default is False.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing the generated embeddings and tokenized text.

    Todo:
        Add an option to cancel embeddings generation after a certain number of windows to ensure
        it finishes within a guaranteed time.

    """
    max_len = get_model_max_len(load_model(model_id=model_id))
    overlap = int(max_len * overlap_ratio)
    tok_wins, toktxt = tokenize_windows(txt, model_id=model_id, max_len=max_len, overlap_ratio=overlap_ratio)
    if longtextcap:
        tok_wins = tok_wins[:100]

    if status_bar:  # only use tqdm for "long lasting" transformations
        vec_wins = [transform_to_contextual_embeddings(win, model_id=model_id)
                    for win in tqdm(tok_wins)]
    else:
        vec_wins = [transform_to_contextual_embeddings(win, model_id=model_id)
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

    return vecs, toktxt


# old name: create_cross_lingual_embeddings
def create_cross_lingual_contextual_embeddings(txt: str, model_id: str, lang: bool = False):
    # Map the token strings to their vocabulary indeces.
    # indexed_tokens = tokenizer.convert_tokens_to_ids(toktxt)
    tokenizer = load_tokenizer(model_id=model_id)
    input_ids_t = torch.tensor([tokenizer.encode(txt)])

    # for one sentence all ids are "1" for two, the first sentence gets "0"
    segments_ids_t = torch.tensor([[1] * input_ids_t.shape[1]])

    # create language IDs
    if lang:
        language_id = tokenizer.lang2id[lang]  # 0
        langs_t = torch.tensor([language_id] * input_ids_t.shape[1])  # torch.tensor([0, 0, 0, ..., 0])
        # We reshape it to be of size (batch_size, sequence_length)
        langs_t = langs_t.view(1, -1)  # is

    model = load_model(model_id=model_id)

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
    load_model.cache_clear()
    load_tokenizer.cache_clear()


def vecseq_similarity(vs, search_vec):
    return cos_compare(vs, [search_vec])


@functools.lru_cache
def get_vocabulary(model_id: str):
    """make sure the vocabulary only gets loaded once
    TODO: implement more vocabularies"""
    logger.info(f"loading vocabulary from {model_id}")
    model = load_model(model_id)
    # return and transform embeddings into numpy array
    return model.embeddings.word_embeddings.weight.detach().numpy()


def fullword_embeddings(toktxt: list[str], vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get embeddings for entire words by sewing word pieces back together.

    Args:
        toktxt (List[str]): Tokenized text as a list of word pieces.
        vs (np.ndarray): Word piece vectors as a 2D numpy array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns two 1D numpy arrays for full word-tokens and vectors respectively.
    """
    # Average embedding vectors for entire words:
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
        # TODO: replace np.mean with np.sum here and below?
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


def get_max_word_similarity(vs, searchstring, model_id: str):
    sv, _ = get_tokenizer_only_embeddings(searchstring, model_id=model_id)
    match = vecseq_similarity(vs, sv.mean(axis=0))
    return match.max()


def search(toktxt, vs, searchstring: str, model_id: str, num=1):
    """
    returns:
        top tokens, token ids, correponding scores, all token scores
    """
    # sv, _ = longtxt_embeddings(search_word,model,tokenizer)
    sv, _ = get_tokenizer_only_embeddings(searchstring, model_id=model_id)

    match = vecseq_similarity(vs, sv.mean(axis=0))
    return top_search_results(toktxt, match, num=num) + (match,)


def calculate_string_embeddings(text: str, model_id: str, only_tokenizer: bool, overlap_ratio: float = 0.1):
    """
    this method converts a text of arbitrary length into
    a vector.
    """
    if only_tokenizer:
        vs, toktxt = get_tokenizer_only_embeddings(text, model_id)
        return vs, toktxt
    else:
        # vs, toktxt = longtxt_embeddings(text, model, tokenizer, np.mean)
        vs, toktxt = longtxt_embeddings(text, model_id, overlap_ratio=overlap_ratio)
        return vs, toktxt


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
    model = load_model(model_id)
    sv, _ = longtxt_embeddings(topic, model)
    sv.mean()
    # sv, _ = get_embeddings(searchstring,tokenizer)

    similarity = cos_compare([vs], [sv])
    # match = vecseq_similarity(vs,sv.mean(axis=0))

    return similarity


# TODO: we need to move this into the nlp_context class
#      and provide the use of the models with a timeout
#      or a "with" context or something similar...

# @functools.lru_cache()


def convert_ids_to_string(model_id: str, ids):
    tokenizer = load_tokenizer(model_id=model_id)
    a = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(a)


@functools.lru_cache
def load_pipeline(pipeline_type: str, model_id: str):
    from transformers import pipeline
    pipeline_instance = pipeline(pipeline_type, model=model_id)
    return pipeline_instance


@functools.lru_cache
def load_qa_models(model_id: str):
    # TODO: only load model "id" and use that id
    #        with transformers AutoModel etc...
    logger.info(f"loading Q & A model and tokenizer {model_id}")
    # model, tokenizer = load_models(model_id)
    # TODO: use load_tokenizer function for this
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    logger.info(f"finished loading Q & A models... {model_id}")
    return model, tokenizer


def get_model_max_len(model):
    try:
        max_input_tokens = model.config.max_position_embeddings
    except AttributeError:
        # TOOD: we should probably find a better method for this....
        max_input_tokens = 512  # e.g. this is a good choice for t5 (doesn't have a hard limit,
        # but huge memory consumption)

    return max_input_tokens


def summarize_long_text(
        long_text: str,
        model_id: str,
        token_overlap=50,
        max_len=200,
):
    """
    This function is rather slow due to the recursive nature of it.
    It is usually a good idea to do some pre-processing of the
    text before using this summarizer. For example reducing the text size
    using a textrank algorithm which filters out unimportant sentences .
    """
    pipeline = load_pipeline("summarization", model_id=model_id)
    model, tokenizer = pipeline.model, pipeline.tokenizer
    max_input_tokens = get_model_max_len(model)

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

    return summary.strip()
