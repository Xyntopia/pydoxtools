from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
from typing import Callable

import requests
import yaml
from diskcache import Cache

from .operators_base import Operator
from .settings import settings

cache = Cache(settings.PDX_CACHE_DIR_BASE / "chat_answers")

import logging

logger = logging.getLogger(__name__)


@cache.memoize()
def openai_chat_completion_with_diskcache(
        model_id: str, temperature: float,
        messages: tuple[dict[str, str], ...],
        max_tokens: int = 256,
):
    import openai  # only import openai if we actually need it :)
    openai.api_key = settings.OPENAI_API_KEY
    completion = openai.ChatCompletion.create(
        model=model_id,
        temperature=temperature,
        messages=messages,
        max_tokens=max_tokens
    )
    return completion


# TODO: add a "normal" prompt

{'prompt': ["Y"],
 'model': 'text-davinci-003',
 'temperature': 0.0,
 'max_tokens': 256,
 'top_p': 1,
 'frequency_penalty': 0,
 'presence_penalty': 0,
 'n': 1,
 'best_of': 1,
 'logit_bias': {}}


@cache.memoize()
def openai_chat_completion(msgs, model_id='gpt-3.5-turbo', max_tokens=256):
    completion = openai_chat_completion_with_diskcache(
        model_id=model_id, temperature=0.0, messages=msgs, max_tokens=max_tokens
    )
    result = completion.choices[0].message
    return result


@functools.lru_cache
def get_cached_gpt4model(model_id):
    from gpt4all import GPT4All
    # gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
    gptj = GPT4All(model_id)
    return gptj


@functools.lru_cache
def gpt4_models():
    try:
        from gpt4all import GPT4All
        return [m["filename"].strip('.bin') for m in GPT4All.list_models()]
    except requests.exceptions.JSONDecodeError:
        return [""]


@cache.memoize()
def gpt4allchat(messages, model_id="ggml-mpt-7b-instruct", max_tokens=256, temperature=0.0):
    model = get_cached_gpt4model(model_id)
    task_msg = "\n".join([m['content'] for m in messages])
    res = model.generate(prompt=task_msg, temp=temperature, max_tokens=max_tokens)

    return res[0]


def chat_completion(msgs: list[dict[str, str], ...], model_id: str, max_tokens: int) -> str:
    if model_id in ["gpt-3.5-turbo", "gpt-4"]:
        completion = openai_chat_completion_with_diskcache(
            model_id=model_id, temperature=0.0, messages=tuple(msgs), max_tokens=max_tokens)
        result = completion.choices[0].message['content']
    elif model_id in gpt4_models():
        completion = gpt4allchat(
            model_id=model_id, temperature=0.0, messages=msgs, max_tokens=max_tokens)
        result = completion  # ["choices"][0]['message']['content']
    else:
        result = ""
        raise NotImplementedError(f"We can currently not handle the model {model_id}")

    return result


def task_chat(
        task,
        context=None,
        previous_tasks=None,
        objective=None,
        format="yaml",
        method="prompt"
):
    """This function generates a chat prompt for a task. It is used to generate the prompt for the chat
    interface"""
    msgs = []
    msgs.append({"role": "system",
                 "content": "You are a helpful assistant that aims to complete the given task."
                            "Do not add any amount of explanatory text."})
    if method == "chat":
        msgs.append({"role": "user",
                     "content": f"# Overall objective: \n{objective}\n\n"}),
        if previous_tasks:
            msgs.append({"role": "user",
                         "content": f"# Take into account these previously completed tasks"
                                    f"\n\n{previous_tasks} \n\n"}),
        if context:
            msgs.append({"role": "user",
                         "content": f"# Take into account this context"
                                    f"\n\n{context} \n\n"}),
        if False:
            msgs.append({"role": "user",
                         "content": f"# Instruction: "
                                    f"The prompt below is a question to answer, a task to complete, or a conversation "
                                    f"to respond to; decide which and write an appropriate response.\n\n"
                                    f"## Prompt: {task} \n\n"
                                    f"## Result:\n\n"})

        if task: msgs.append({"role": "user",
                              "content": f"# Complete the following task: \n{task} \n\n"
                                         f"Provide only the precise information requested without context, "
                                         f"make sure we can parse the response as {format}. \n\n"
                                         f"## RESULT:\n"})
    else:
        msg = f"# Considering the overall objective: \n{objective}\n\n"
        if previous_tasks:
            msg += f"# Take into account these previously completed tasks:\n\n{previous_tasks} \n\n"
        if context:
            msg += f"# Take into account this context:\n\n{context} \n\n"
        msg += f"# Complete the following task: \n{task} \n\n" \
               f"Provide only the precise information requested without context, " \
               f"make sure we can parse the response as {format}. RESULT:\n"
        msgs.append(
            {"role": "user", "content": msg})
    return msgs


def execute_task(task, previous_tasks=None, context=None, objective=None,
                 formatting="yaml", model_id="ggml-mpt-7b-instruct",
                 max_tokens=1000):
    """Creates a message and executes a task with an LLM based on given information"""
    msgs = task_chat(previous_tasks=previous_tasks,
                     context="\n---\n".join(context) if context else None, task=task,
                     objective=objective,
                     format=formatting)
    res = chat_completion(msgs, max_tokens=max_tokens, model_id=model_id)
    if formatting == "yaml":
        try:
            obj = yaml_loader(res)
        except yaml.YAMLError:
            raise yaml.YAMLError(f"Could not convert {res} to yaml.")
    elif formatting == "txt":
        obj = res
    elif formatting == "markdown":
        obj = res
    else:
        logger.warning(f"Formatting: {formatting} is unknown!")
        pass  # do nothing ;)

    return obj, msgs, res


def safe_execute_task(*args, **kwargs) -> tuple:
    try:
        res = execute_task(*args, **kwargs)
        # res,msgs,txt = execute_task(task, model_id="gpt-4")
    except yaml.YAMLError as e:
        logger.info(e)
        res = None, None, None

    return res


class LLMChat(Operator):
    """
    Use LLMChat on data in our pipeline!

    model_id: if model_language=="auto" we also need to set our model_size
    """

    # TODO: add a "HuggingfaceExtractor" with similar structure
    def __call__(
            self, property_dict: Callable, model_id: str
    ) -> Callable[[list[str], list[str] | str], list[str]]:
        def task_machine(
                tasks: list[str],
                props: list[str] | str = "full_text"
        ) -> list[str]:
            # choose the property that we want to ask a question about...
            if isinstance(props, str):
                # if we only have a single variable, use the data directly...
                data = property_dict(props)[props]
            else:
                data = property_dict(*props)

            text = yaml.dump(data)

            if isinstance(tasks, str):
                tasks = [str]

            results = []
            # create a completion
            for task in tasks:
                msgs = ({"role": "system",
                         "content": "You are a helpful assistant that aims to complete the given task."
                                    "Do not add any amount of explanatory text.\n"},
                        {"role": "user",
                         "content": f"# Instruction: "
                                    f"The prompt below is a question to answer, a task to complete, or a conversation "
                                    f"to respond to; decide which and write an appropriate response.\n\n"
                                    f"## Prompt:\n\n{task}\n\n"
                                    f"## Input for the task:\n\n{text}.\n\n"
                                    f"## Result:\n\n"})

                result = chat_completion(msgs, model_id=model_id, max_tokens=2000)

                results.append(result)
            return results

        # list models
        # models = openai.Model.list()

        # print the first model's id
        # print(models.data[0].id)

        return task_machine


def very_slow():
    from transformers import AutoTokenizer
    import transformers
    import torch

    model = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def yaml_loader(txt: str):
    # Remove ```yaml and ``` if they are present in the string
    txt = txt.strip().replace("```yaml", "").replace("```", "")
    txt = txt.strip("`")
    data = yaml.unsafe_load(txt)
    return data
