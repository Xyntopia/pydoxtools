from pydoxtools import document


class Alias(document.Extractor):
    """Connect extractor variables with Aliases"""

    def __call__(self, **kwargs):
        return kwargs


class LambdaExtractor(document.Extractor):
    """Wrap an arbitrary function as an Extractor"""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
