import openai
from typing import Callable
import yaml

from .document_base import Extractor
from .settings import settings


class OpenAIChat(Extractor):
    def __init__(
            self,
            # TODO: generalize this in order to support e.g. alpaca
            # service = "openai"
            # model = "XXXX"
    ):
        """
        model_size: if model_language=="auto" we also need to set our model_size

        TODO: add a "HuggingfaceExtractor" with similar structure

        """
        super().__init__()

    def __call__(self, property_dict: Callable, model_id: str = None):
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
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that aims to complete the given task."
                                    "Do not add any amount of explanatory text."},
                        {"role": "user", "content": f"# Instruction: {task} ## Input for the task: {text}."}]
                )
                result = completion.choices[0].message
                results.append(result)
            return results

        # list models
        # models = openai.Model.list()

        # print the first model's id
        # print(models.data[0].id)

        return task_machine
