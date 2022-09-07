from pydoxtools import document


class LambdaExtractor(document.Extractor):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
