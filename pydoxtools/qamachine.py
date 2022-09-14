#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import logging
from typing import Dict, List, Tuple, Any

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from pydoxtools.document import Extractor
from pydoxtools.nlp_utils import tokenize_windows

logger = logging.getLogger(__name__)


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


def answer_questions_on_long_text(questions, text, nlp_context) -> Dict[str, List[Tuple[str, float]]]:
    all_answers = {}
    for q in questions:
        answers = long_text_question(
            q, text, nlp_context.tokenizer, nlp_context.model)
        all_answers[q] = answers

    return all_answers


# we have functools.lru_cache outside of NLPContext because we would like to
# cache this function and there is memory leak when using lru_cache on a member function
# https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object
@functools.lru_cache()
def QandAmodels(model_type="fast"):
    # TODO: only load model "id" and use that id
    #        with transformers AutoModel etc...
    logger.info("loading Q & A model and tokenizer")

    if model_type == "slow":
        # also very good, but slow:
        model_name = 'replydotai/albert-xxlarge-v1-finetuned-squad2'
    elif model_type == "medium":
        # very good and fast:
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        # model_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'
    elif model_type == "multi":
        # 'mrm8488/distilbert-multi-finedtuned-squad-pt'
        model_name = 'mrm8488/bert-multi-uncased-finetuned-xquadv1'
    elif model_type == 'base':
        model_name = 'deepset/bert-base-cased-squad2'
    elif model_type == 'large':
        model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
    elif model_type == 't5':
        model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    elif model_type == "fast":
        # not very good at this task, but fast
        # distilbert-base-cased-distilled-squad
        # model_name = 'mrm8488/bert-small-finetuned-squadv2'
        model_name = 'distilbert-base-cased-distilled-squad'
        # model_name = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    else:
        model_name = model_type

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    logger.info("finished loading Q & A models...")
    return NLPContext(tokenizer=tokenizer, model=model)


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


def long_text_question(question, text, tokenizer, model):
    max_len = 512  # maximum possble input for BERT and other transformers
    q_inputs = tokenizer(question, add_special_tokens=False, return_tensors="pt")
    q_len = q_inputs.input_ids.shape[1]
    max_txt_len = max_len - q_len - 3  # -1, because the [CLS] token from q_len will get truncated
    win_tokens, win_toktxt = tokenize_windows(
        text, tokenizer, win_len=max_txt_len, overlap=50, max_len=512,
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


class QamExtractor(Extractor):
    """
    Question Asnwering Machine Extractor

    The Extractor takes questions and gives back answers on a text.
"""

    def __init__(self, model_type="medium"):
        super().__init__()
        self._model_type = model_type

    def __call__(self, questions: list[str], text: str):
        nlpc = QandAmodels(self._model_type)
        allanswers = answer_questions_on_long_text(questions, text, nlpc)
        answers = list(allanswers.values())
        return answers


if __name__ == "__main__":
    # short test if NLP functions are working the way the're supposed to...

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
