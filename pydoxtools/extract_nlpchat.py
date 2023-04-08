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

    def __call__(self, , tasks: str, full_text: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = settings.OPENAI_API_KEY

        def task_machine(tasks: list[str]) -> str:
            if isinstance(tasks, str):
                tasks = [str]

            # create a completion
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"# Instruction: {task} ## Input for the task: {full_text}"}]
            )
            result = completion.choices[0].message
            return result

        # list models
        # models = openai.Model.list()

        # print the first model's id
        # print(models.data[0].id)

        return task_machine
