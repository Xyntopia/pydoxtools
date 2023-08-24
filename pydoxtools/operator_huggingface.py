#!/usr/bin/env python3
"""
Huggingface related code

Enables loading of mdoels from huggingface for all kinds of purposes
"""

from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import logging
from typing import Dict, List, Tuple, Callable

import torch

from pydoxtools import nlp_utils
from pydoxtools.list_utils import ensure_list
from pydoxtools.nlp_utils import tokenize_windows
from pydoxtools.operators_base import Operator

logger = logging.getLogger(__name__)


def answer_questions_on_long_text(questions, text, model_id) -> Dict[str, List[Tuple[str, float]]]:
    all_answers = {}
    for q in questions:
        answers = long_text_question(
            q, text, model_id=model_id)
        all_answers[q] = sorted(answers, key=lambda x: -x[1])

    return all_answers


# we have functools.lru_cache outside of NLPContext because we would like to
# cache this function and there is memory leak when using lru_cache on a member function
# https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object


def convert_id_range_to_text(input_ids, answer_start, answer_end, tokenizer):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )


def get_topk_answers(answer_start_scores, answer_end_scores,
                     input_ids, ans_num, tokenizer,
                     max_ans_words=5, max_ans_len=100):
    # answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    if ans_num > 5:
        raise NotImplementedError("k can not be > 5.")
    answers_start = torch.topk(answer_start_scores, k=5)
    answers_start_idx = answers_start.indices.flatten()
    answers_start_scores = answers_start.values.flatten()
    answers_end = torch.topk(answer_end_scores, k=5)
    answers_end_idx = answers_end.indices.flatten()
    answers_end_scores = answers_end.values.flatten()
    # TODO: get more results by calculating 2D-grid over asnwer_start/end scores
    answers = []
    for i in range(ans_num):
        answer = convert_id_range_to_text(
            input_ids, answers_start_idx[i], answers_end_idx[i] + 1, tokenizer)
        if answer in ['', '[CLS]', '[SEP]', '[unused1]']:
            continue
        if '[unused2]' in answer:
            continue
        if '[unused3]' in answer:
            continue
        if len(answer) > max_ans_len:
            continue
        if len(answer.split()) > max_ans_words:
            continue
        answer_score = (answers_start_scores[i] + answers_end_scores[i]).detach().numpy()
        answers.append((answer, answer_score.item()))
    return answers


def long_text_question(question, text, model_id: str):
    max_len = 512  # maximum possble input for BERT and other transformers
    model, tokenizer = nlp_utils.load_qa_models(model_id=model_id)
    q_inputs = tokenizer(question, add_special_tokens=False, return_tensors="pt")
    q_len = q_inputs.input_ids.shape[1]
    max_txt_len = max_len - q_len - 3  # -1, because the [CLS] token from q_len will get truncated
    win_tokens, win_toktxt = tokenize_windows(
        text, model_id, win_len=max_txt_len, overlap_ratio=0.1, max_len=512,
        add_special_tokens=False)

    q_tokens = q_inputs.input_ids[0].tolist()
    q_len = len(q_tokens)
    answers = []
    # check if model needs type_ids:
    needs_type_ids = 'token_type_ids' in tokenizer("test1", "test2")
    for i, txt_tokens in enumerate(win_tokens):
        logger.info(f"run q & a segment #: {i}")
        input_ids = torch.tensor([[2] + q_tokens + [3] + txt_tokens + [3]])
        input = dict(
            input_ids=input_ids,
            attention_mask=torch.tensor([input_ids.shape[1] * [1]])
        )
        if needs_type_ids:
            input["token_type_ids"] = torch.tensor([[0] * (q_len + 2) + [1] * (len(txt_tokens) + 1)])
        res = model(**input)
        answer_start_scores, answer_end_scores = res.start_logits, res.end_logits
        answers.extend(get_topk_answers(
            answer_start_scores, answer_end_scores, input_ids[0], 5, tokenizer))

    return answers


def question_text_segment(text, question, tokenizer, model, ans_num=1):
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    answers = get_topk_answers(*model(**inputs), input_ids, 5)
    return answers


class HuggingfacePipeline(Operator):
    """
    Configurable Operator to load models from huggingface for various
    tasks:  Summarization, Question answering etc..

    We can run this on every property in a document.
    """

    def __init__(self, pipeline: str):
        super().__init__()
        self._pipeline = pipeline

    def __call__(self, property_dict: Callable, trf_model_id: str):
        pipeline = nlp_utils.load_pipeline(self._pipeline, model_id=trf_model_id)

        @functools.lru_cache
        def inference(
                input: list[str] | str,
                props: list[str] | str = "full_text"
        ) -> list[list[tuple[str, float]]]:
            # choose the property that we want to ask a question about...
            if isinstance(props, str):
                # if we only have a single variable, use the data directly...
                data = property_dict(props)[props]
            else:
                data = property_dict(*props)

            text = str(data)
            input = ensure_list(input)

            res = pipeline(input, context=text)
            return res

        return inference


class QamExtractor(Operator):
    """
    Question Answering Machine Operator

    The Operator generates a function takes questions and gives back
    answers on the given text.
"""

    def __call__(self, property_dict: Callable, trf_model_id: str = None):
        @functools.lru_cache
        def qa_machine(
                questions: list[str] | str,
                props: list[str] | str = "full_text"
        ) -> list[list[tuple[str, float]]]:
            # choose the property that we want to ask a question about...
            if isinstance(props, str):
                # if we only have a single variable, use the data directly...
                data = property_dict(props)[props]
            else:
                data = property_dict(*props)

            # TODO: sort answers regarding threshold
            text = str(data)
            questions = ensure_list(questions)
            allanswers = answer_questions_on_long_text(questions, text, model_id=trf_model_id)
            answers = list(allanswers.values())
            return answers

        return qa_machine


if __name__ == "__main__":
    # short test if NLP functions are working the way the're supposed to...
    # TODO: move this into tests...

    short_test_text = """
    The Eli-2020 is a highly enjoyable therapy gadget. It features four limbs and 
    can occasionally smile at you causing the desired relaxing effect.

    It needs to be energized at least every 2 hours or so, but occasionally it
    makes sense to just leave it on the plug. It will automatically stop recharging
    once it is full.
    """

    logging.basicConfig(level=logging.INFO)

    f = "/home/tom/comcharax/data/pdfs/wwwmssscom/c50.pdf"

    questions = [
        "What is the product name?",
        "What are the products features?",
        "What is the purpose of it?"
    ]
    text = short_test_text
    # text = f"\n\n{'-'*10}\n\n".join(pdf_text['textboxes'])

    nlpc = QandAmodels()
    allanswers = answer_questions_on_long_text(questions, text, nlpc)

    logger.info(allanswers)
    # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # these are possible alternativ methods to extract answers:
    """
    if False:
        from transformers import DistilBertTokenizer, DistilBertForTokenClassification, AutoTokenizer, \
            AutoModelForQuestionAnswering
        import torch

        # token classification (NER)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', return_dict=True)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

    if False:
        importance = veclengths(vs)

        searchstring = 'item with wifi'
        (best_matches,
         idxs,
         scores, match) = search(toktxt, vs, searchstring, model, tokenizer, 10)
        colhtml = html_utils.color_text(toktxt[:], match)
        oib(colhtml)
    """
