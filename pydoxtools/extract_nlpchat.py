import openai

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

    def __call__(self, full_text: str, model_id: str = None):
        openai.api_key = settings.OPENAI_API_KEY

        def task_machine(tasks: list[str]) -> list[str]:
            if isinstance(tasks, str):
                tasks = [str]

            results = []
            # create a completion
            for task in tasks:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that aims to complete the given task"
                                                      "with as little text as possible."},
                        {"role": "user", "content": f"# Instruction: {task} ## Input for the task: {full_text}."}]
                )
                result = completion.choices[0].message
                results.append(result)
            return results

        # list models
        # models = openai.Model.list()

        # print the first model's id
        # print(models.data[0].id)

        return task_machine
