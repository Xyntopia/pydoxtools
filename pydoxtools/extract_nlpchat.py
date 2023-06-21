from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
from typing import Callable
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import yaml
from diskcache import Cache

from .operators_base import Operator
from .settings import settings

cache = Cache(settings.PDX_CACHE_DIR_BASE / "chat_answers")


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
def get_gpt4model(model_id):
    from gpt4all import GPT4All
    # gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
    gptj = GPT4All(model_id)
    return gptj


@functools.lru_cache
def gpt4_models():
    from gpt4all import GPT4All
    return [m["filename"].strip('.bin') for m in GPT4All.list_models()]


@functools.lru_cache
def gpt4allchat(messages, model_id="ggml-mpt-7b-instruct", max_tokens=256, *args, **kwargs):
    gptj = get_gpt4model(model_id)
    res = gptj.chat_completion(messages, default_prompt_footer=False, default_prompt_header=False)
    return res


def chat_completion(msgs: tuple[dict[str, str], ...], model_id: str) -> str:
    if model_id == "gpt-3.5-turbo":
        completion = openai_chat_completion_with_diskcache(
            model_id=model_id, temperature=0.0, messages=msgs
        )
        result = completion.choices[0].message
    elif model_id in gpt4_models():
        completion = gpt4allchat(
            model_id=model_id, temperature=0.0, messages=msgs
        )
        result = completion["choices"][0]['message']['content']
    else:
        result = ""
        NotImplementedError(f"We can currently not handle the model {model_id}")

    return result


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
                                    "Do not add any amount of explanatory text."},
                        {"role": "user",
                         "content": f"# Instruction: "
                                    f"The prompt below is a question to answer, a task to complete, or a conversation "
                                    f"to respond to; decide which and write an appropriate response.\n\n"
                                    f"## Prompt: {task} \n\n"
                                    f"## Input for the task: {text}.\n\n"
                                    f"## Result:\n\n"})

                result = chat_completion(msgs, model_id=model_id)

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


def reasonable_speed():
    from gpt4all import GPT4All

    # gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
    gptj = GPT4All("ggml-mpt-7b-instruct")
    messages = [{"role": "user", "content": "Name 3 colors"}]
    res = gptj.chat_completion(messages)

    messages = [{"role": "user", "content": "What colors did you previously name?"}]
    gptj.chat_completion(messages)

    messages = [
        {"role": "user", "content": "Name 3 colors"},
        {'role': 'assistant', 'content': 'Blue, Green and Yellow'},
        {"role": "user", "content": "What colors did you previously name?"}]
    gptj.chat_completion(messages)
