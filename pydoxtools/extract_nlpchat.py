from typing import Callable

import openai
import yaml
from diskcache import Cache

from .operators_base import Operator
from .settings import settings

cache = Cache(settings.PDX_CACHE_DIR_BASE / "openai")


@cache.memoize()
def openai_chat_completion_with_diskcache(
        model_id: str, temperature: float,
        messages: tuple[dict[str, str], ...],
        max_tokens: int = 256,
):
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


class OpenAIChat(Operator):
    """
    Use OpenAIChat on data in our pipeline!

    model_id: if model_language=="auto" we also need to set our model_size
    """

    # TODO: add a "HuggingfaceExtractor" with similar structure
    def __call__(
            self, property_dict: Callable, model_id: str
    ) -> Callable[[list[str], list[str] | str], list[str]]:
        # TODO: move this into a more generic place...
        openai.api_key = settings.OPENAI_API_KEY

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
                        {"role": "user", "content": f"# Instruction: {task} \n\n"
                                                    f"## Input for the task: {text}.\n\n"
                                                    f"## Output:\n\n"})
                completion = openai_prompt_with_diskcache(
                    model_id=model_id, temperature=0.0, messages=msgs
                )
                result = completion.choices[0].message
                results.append(result)
            return results

        # list models
        # models = openai.Model.list()

        # print the first model's id
        # print(models.data[0].id)

        return task_machine
